import os
import re
import csv
import json
import time
import glob
import pickle
import random
import itertools
from datetime import datetime
from collections import defaultdict, deque

import requests
from bs4 import BeautifulSoup

import numpy as np
import pandas as pd
import torch
import tiktoken
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

import openai
from openai import AzureOpenAI, OpenAIError, RateLimitError
import google.generativeai as genai

from bert_score import score as bert_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
from sklearn.utils.multiclass import unique_labels

from src.utils.file_utils import zip_directory
from src.preprocessing.nlp.web_scraping_utils import *
from src.preprocessing.nlp.pdf_preprocessing import (
    extract_text_from_pdf,
    save_to_json,
    process_multiple_pdfs
)
from src.nlp.utils.text_cleaning import (
    clean_text,
    is_navigation_item,
    clean_website_data,
    clean_pdf_data,
    clean_list_based_json,
    process_json_files
)
from src.nlp.api.config import openai_client, embedding_function, genai
from src.nlp.rag.chromadb import (
    store_embeddings_in_chroma,
    process_and_store_embeddings,
    store_embeddings_in_chroma_openai,
    process_and_store_embeddings_openai
)
from src.nlp.rag.embeddings_utils import structure_documents, load_all_embeddings
from src.nlp.rag.embeddings import GeminiEmbeddings, MiniLMEmbeddings, get_embedding
from src.nlp.rag.embeddings_openai import structure_documents_openai, get_embedding_openai
from src.nlp.rag.retrieval import retrieve_top_k_chromadb, hybrid_retrieval, combined_retrieval
from src.nlp.prompt.prompt import generate_structured_prompt_tnm
from src.nlp.api.gpt4omini import (
    get_azure_openai_rate_limits,
    enforce_rate_limits_openai,
    generate_response_gpt4o
)
from src.nlp.api.gemini_2flash import generate_response_gemini
from src.nlp.rag.chunking import chunk_text, chunk_text_openai
from src.nlp.rag.rag_pipeline import retrieval_and_response_pipeline



def enforce_rate_limits_eval(token_count, llm_model):
    """Enforce rate limits dynamically for GPT models only."""
    global REQUEST_TIMESTAMPS, TOKENS_USED
    current_time = time.time()

    if "gemini" in llm_model.lower():
        # Gemini: do not apply rate limit logic here, let exception handling manage it
        return

    # GPT-style rate limiting
    while REQUEST_TIMESTAMPS and (current_time - REQUEST_TIMESTAMPS[0]) > 60:
        REQUEST_TIMESTAMPS.popleft()

    requests_remaining = MAX_REQUESTS_PER_MINUTE - len(REQUEST_TIMESTAMPS)
    tokens_remaining = MAX_TOKENS_PER_MINUTE - TOKENS_USED

    if requests_remaining <= 0 or tokens_remaining < token_count:
        if REQUEST_TIMESTAMPS:
            sleep_time = max(1, 60 - (current_time - REQUEST_TIMESTAMPS[0]))
        else:
            sleep_time = 60
        print(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds.")
        time.sleep(sleep_time)
        TOKENS_USED = 0

    REQUEST_TIMESTAMPS.append(current_time)
    TOKENS_USED += token_count


def compute_bertscore_in_batches(preds, refs, batch_size=100, lang="en"):
    f1_all = []
    for i in tqdm(range(0, len(preds), batch_size), desc="BERTScore batches"):
        batch_preds = preds[i:i+batch_size]
        batch_refs = refs[i:i+batch_size]
        try:
            _, _, f1 = bert_score(batch_preds, batch_refs, lang=lang, verbose=False)
            f1_all.extend(f1.cpu().numpy())
        except Exception as e:
            print(f"[BERTScore Error at batch {i}] {e}")
            f1_all.extend([np.nan] * len(batch_preds))  # fallback in case of failure
    return f1_all

def compute_ragas_safe(dataset):
    try:
        return evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=llm,
            embeddings=embeddings
        )
    except Exception as e:
        print(f"[RAGAS error] {e}")
        return {"faithfulness": [np.nan], "answer_relevancy": [np.nan],
                "context_precision": [np.nan], "context_recall": [np.nan]}


def chunk_dataset(dataset, chunk_size):
    for i in range(0, len(dataset), chunk_size):
        yield dataset.select(range(i, min(i + chunk_size, len(dataset))))

def find_ground_truth(test_set, t_stage, n_stage, m_stage, age, cancer_type):
    """
    Finds the ground truth stage and treatment for the given patient based on NSCLC or SCLC rules.
    """
    if "SMALL CELL" in cancer_type.upper():
        for _, row in test_set[test_set["Cancer Type"].str.upper().str.contains("SMALL CELL")].iterrows():
            t_vals = [v.strip() for v in row["T-Stage"].split(";")]
            n_vals = [v.strip() for v in row["N-Stage"].split(";")]
            m_vals = [v.strip() for v in row["M-Stage"].split(";")]
            age_label = row["Age"].strip()

            t_match = "Any T" in t_vals or t_stage in t_vals
            n_match = "Any N" in n_vals or n_stage in n_vals
            m_match = "Any M" in m_vals or m_stage in m_vals

            age_group = ">= 70" if age >= 70 else "< 70"

            if t_match and n_match and m_match and age_group == age_label:
                return row["Stage"], row["Treatment Plan"]
    else:
        match = test_set[
            (test_set["T-Stage"] == t_stage) &
            (test_set["N-Stage"] == n_stage) &
            (test_set["M-Stage"] == m_stage)
        ]

        if match.empty and m_stage.startswith("M1"):
            match = test_set[
                (test_set["M-Stage"].str.startswith("M1")) &
                ((test_set["T-Stage"] == "Any T") | test_set["T-Stage"].str.contains("Any", case=False)) &
                ((test_set["N-Stage"] == "Any N") | test_set["N-Stage"].str.contains("Any", case=False))
            ]

        if not match.empty:
            return match.iloc[0]["Stage"], match.iloc[0]["Treatment Plan"]

    return None, None

def custom_evaluation_pipeline(configurations, k_range, temperature_range, n_repeats,
                                   output_path):
    test_set = load_ground_truth()
    patient_inputs = load_patient_inputs()
    all_results = []
    ragas_records_map = defaultdict(list)

    for idx, patient in tqdm(patient_inputs.iterrows(), total=len(patient_inputs), desc="Patients"):
        t_stage, n_stage, m_stage = patient["T_stage"], patient["N_stage"], patient["M_stage"]
        age, cancer_type, gender, smoker = int(patient["age"]), patient["cancer_type"], patient["Gender"], patient["smoking_status"]
        additional_info = f"Smoker: {'Yes' if smoker else 'No'}"
        true_stage, true_treatment = find_ground_truth(test_set, t_stage, n_stage, m_stage, age, cancer_type)
        if not true_stage:
            continue

        for config in tqdm(configurations, desc="Configurations", leave=False):
            for top_k, temperature in itertools.product(k_range, temperature_range):
                for run_idx in range(n_repeats):
                    max_retries = 2
                    retry_delay = 15  # segundos

                    success = False
                    for attempt in range(max_retries):
                        # Rate limit logic before calling the API
                        try:
                            enforce_rate_limits_eval(6000, config["llm_model"])
                            response, retrieved_docs = retrieval_and_response_pipeline(
                                query="Based on the patient data and TMN staging what is the exact stage of the cancer and the indicated course of treatment?",
                                embedding_model=config["embedding_model"],
                                retrieval_method=config["retrieval_method"],
                                llm_model=config["llm_model"],
                                t_stage=t_stage,
                                n_stage=n_stage,
                                m_stage=m_stage,
                                histopath_grade="",
                                cancer_type=cancer_type,
                                age=age,
                                gender=gender,
                                additional_info=additional_info,
                                top_k=top_k
                            )
                            if response and not str(response).startswith("ERROR"):
                                success = True
                                break

                        except Exception as e:
                            error_message = str(e)
                            if "429" in error_message and "gemini" in config["llm_model"].lower():
                                print(" Gemini quota exceeded. Sleeping for 24 hours.")
                                time.sleep(86410)
                            else:
                                print(f"[Attempt {attempt+1}/{max_retries}] API Error: {e}")

                            print(f"[Attempt {attempt+1}/{max_retries}] Retrying in {retry_delay} seconds...")
                            time.sleep(retry_delay)

                    if not success:
                        print(f"F ailed to get response after {max_retries} retries. Skipping input_idx={idx}")
                        continue  # Skip this run_idx safely



                    pred_stage, pred_treatment = extract_stage_and_treatment(response)
                    result = {
                        **config, "top_k": top_k, "temperature": temperature,
                        "input_idx": idx, "run_idx": run_idx,
                        "T_stage": t_stage, "N_stage": n_stage, "M_stage": m_stage,
                        "cancer_type": cancer_type, "true_stage": true_stage,
                        "predicted_stage": pred_stage, "ground_truth_answer": true_treatment,
                        "predicted_treatment": pred_treatment, "raw_output": response,
                        "contexts": retrieved_docs
                    }
                    all_results.append(result)
                    key = (config["embedding_model"], config["retrieval_method"], config["llm_model"])
                    ragas_records_map[key].append({
                        "question": result["raw_output"],
                        "contexts": result["contexts"],
                        "answer": result["predicted_treatment"],
                        "ground_truth": result["ground_truth_answer"]
                    })

    results_df = pd.DataFrame(all_results)
    results_df["stage_match"] = results_df["true_stage"] == results_df["predicted_stage"]
    
    results_df.to_csv(output_path, index=False)
    print(f"[✔] Saved raw results (before metrics) to {output_path}")
    
def evaluate_from_csv_only(input_csv_path, output_path, grouped_output_path,
                           metrics_output_path, classification_report_output_path,
                           confusion_matrices_csv_path, ragas_output_path):
    import pandas as pd
    from datasets import Dataset
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from ragas import evaluate
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score
    from sklearn.utils.multiclass import unique_labels
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
    from collections import defaultdict
    import numpy as np
    from tqdm import tqdm

    df = pd.read_csv(input_csv_path)
    df["stage_match"] = df["true_stage"] == df["predicted_stage"]

    # Compute BERTScore
    preds = df["raw_output"].fillna("").tolist()
    refs = df["ground_truth_answer"].fillna("").tolist()
    f1_scores = compute_bertscore_in_batches(preds, refs)
    df["bertscore_f1"] = f1_scores

    # Compute BLEU and ROUGE-L
    smoother = SmoothingFunction()
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    bleu_scores, rouge_l_scores = [], []

    for pred, ref in tqdm(zip(preds, refs), total=len(preds), desc="Scoring"):
        bleu = sentence_bleu([ref.split()], pred.split(), smoothing_function=smoother.method1)
        rouge_l = rouge.score(ref, pred)["rougeL"].fmeasure
        bleu_scores.append(bleu)
        rouge_l_scores.append(rouge_l)

    df["bleu"] = bleu_scores
    df["rougeL"] = rouge_l_scores

    df.to_csv(output_path, index=False)

    # Grouped metrics
    grouped = df.groupby(["embedding_model", "retrieval_method", "llm_model"])[
        ["stage_match", "bertscore_f1", "bleu", "rougeL"] #, "top_k", "temperature"
    ].agg(["mean", "std"]).reset_index()
    grouped.columns = ["_".join(col).strip() if isinstance(col, tuple) else col for col in grouped.columns.values]
    grouped.to_csv(grouped_output_path, index=False)

    # Confusion matrix and classification report
    metrics = []
    classification_reports = []
    confusion_matrices = []

    labels = unique_labels(df["true_stage"], df["predicted_stage"])

    for name, group in tqdm(df.groupby(["embedding_model", "retrieval_method", "llm_model"]), desc="Metrics"): #, "top_k", "temperature"
        cm = confusion_matrix(group["true_stage"], group["predicted_stage"], labels=labels)
        acc = accuracy_score(group["true_stage"], group["predicted_stage"])
        f1 = f1_score(group["true_stage"], group["predicted_stage"], average='macro', zero_division=0)
        prec = precision_score(group["true_stage"], group["predicted_stage"], average='macro', zero_division=0)
        rec = recall_score(group["true_stage"], group["predicted_stage"], average='macro', zero_division=0)

        report = classification_report(group["true_stage"], group["predicted_stage"], output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).transpose()
        for i, val in zip(["embedding_model", "retrieval_method", "llm_model"], name): #, "top_k", "temperature"
            report_df[i] = val
        classification_reports.append(report_df)

        metrics.append({
            "embedding_model": name[0],
            "retrieval_method": name[1],
            "llm_model": name[2],
            #"top_k": name[3],
            #"temperature": name[4],
            "accuracy": acc,
            "f1_score": f1,
            "precision": prec,
            "recall": rec,
            "confusion_matrix": cm.tolist()
        })

        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        cm_df.index.name = 'True_Stage'
        cm_df["embedding_model"] = name[0]
        cm_df["retrieval_method"] = name[1]
        cm_df["llm_model"] = name[2]
        #cm_df["top_k"] = name[3]
        #cm_df["temperature"] = name[4]
        confusion_matrices.append(cm_df)

    pd.DataFrame(metrics).to_csv(metrics_output_path, index=False)
    pd.concat(classification_reports).reset_index().to_csv(classification_report_output_path, index=False)
    pd.concat(confusion_matrices).to_csv(confusion_matrices_csv_path, index=True)

    # Compute RAGAS
    print("\n[✔] Computing RAGAS...")
    ragas_summary = []
    grouped = df.groupby(["embedding_model", "retrieval_method", "llm_model"]) #, "top_k", "temperature"

    for name, group in grouped:
        records = []
        for _, row in group.iterrows():
            records.append({
                "question": "Based on the patient data and TMN staging what is the exact stage of the cancer and the indicated course of treatment?",
                "contexts": eval(row["contexts"]) if isinstance(row["contexts"], str) else row["contexts"],
                "answer": row["raw_output"],
                "ground_truth": row["ground_truth_answer"]
            })

        dataset = Dataset.from_list(records)
        ragas_result = compute_ragas_safe(dataset)

        ragas_summary.append({
            "embedding_model": name[0],
            "retrieval_method": name[1],
            "llm_model": name[2],
            #"top_k": name[3],
            #"temperature": name[4],
            "RAGAS_Faithfulness": np.nanmean(ragas_result["faithfulness"]),
            "RAGAS_AnswerRelevancy": np.nanmean(ragas_result["answer_relevancy"]),
            "RAGAS_ContextPrecision": np.nanmean(ragas_result["context_precision"]),
            "RAGAS_ContextRecall": np.nanmean(ragas_result["context_recall"])
        })

    pd.DataFrame(ragas_summary).to_csv(ragas_output_path, index=False)
    print("[✔] Finished evaluation from CSV only.")
    
def evaluate_from_csv(input_csv_path, output_path, grouped_output_path,
                           metrics_output_path, classification_report_output_path,
                           confusion_matrices_csv_path, ragas_output_path):
    import pandas as pd
    from datasets import Dataset
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from ragas import evaluate
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score
    from sklearn.utils.multiclass import unique_labels
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
    from collections import defaultdict
    import numpy as np
    from tqdm import tqdm

    df = pd.read_csv(input_csv_path)
    df["stage_match"] = df["true_stage"] == df["predicted_stage"]

    # Compute BERTScore
    preds = df["raw_output"].fillna("").tolist()
    refs = df["ground_truth_answer"].fillna("").tolist()
    f1_scores = compute_bertscore_in_batches(preds, refs)
    df["bertscore_f1"] = f1_scores

    # Compute BLEU and ROUGE-L
    smoother = SmoothingFunction()
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    bleu_scores, rouge_l_scores = [], []

    for pred, ref in tqdm(zip(preds, refs), total=len(preds), desc="Scoring"):
        bleu = sentence_bleu([ref.split()], pred.split(), smoothing_function=smoother.method1)
        rouge_l = rouge.score(ref, pred)["rougeL"].fmeasure
        bleu_scores.append(bleu)
        rouge_l_scores.append(rouge_l)

    df["bleu"] = bleu_scores
    df["rougeL"] = rouge_l_scores

    df.to_csv(output_path, index=False)

    # Grouped metrics
    grouped = df.groupby(["embedding_model", "retrieval_method", "llm_model", "top_k", "temperature"])[
        ["stage_match", "bertscore_f1", "bleu", "rougeL"] #, "top_k", "temperature"
    ].agg(["mean", "std"]).reset_index()
    grouped.columns = ["_".join(col).strip() if isinstance(col, tuple) else col for col in grouped.columns.values]
    grouped.to_csv(grouped_output_path, index=False)

    # Confusion matrix and classification report
    metrics = []
    classification_reports = []
    confusion_matrices = []

    labels = unique_labels(df["true_stage"], df["predicted_stage"])

    for name, group in tqdm(df.groupby(["embedding_model", "retrieval_method", "llm_model", "top_k", "temperature"]), desc="Metrics"): #, "top_k", "temperature"
        cm = confusion_matrix(group["true_stage"], group["predicted_stage"], labels=labels)
        acc = accuracy_score(group["true_stage"], group["predicted_stage"])
        f1 = f1_score(group["true_stage"], group["predicted_stage"], average='macro', zero_division=0)
        prec = precision_score(group["true_stage"], group["predicted_stage"], average='macro', zero_division=0)
        rec = recall_score(group["true_stage"], group["predicted_stage"], average='macro', zero_division=0)

        report = classification_report(group["true_stage"], group["predicted_stage"], output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).transpose()
        for i, val in zip(["embedding_model", "retrieval_method", "llm_model", "top_k", "temperature"], name): #, "top_k", "temperature"
            report_df[i] = val
        classification_reports.append(report_df)

        metrics.append({
            "embedding_model": name[0],
            "retrieval_method": name[1],
            "llm_model": name[2],
            "top_k": name[3],
            "temperature": name[4],
            "accuracy": acc,
            "f1_score": f1,
            "precision": prec,
            "recall": rec,
            "confusion_matrix": cm.tolist()
        })

        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        cm_df.index.name = 'True_Stage'
        cm_df["embedding_model"] = name[0]
        cm_df["retrieval_method"] = name[1]
        cm_df["llm_model"] = name[2]
        cm_df["top_k"] = name[3]
        cm_df["temperature"] = name[4]
        confusion_matrices.append(cm_df)

    
    pd.DataFrame(metrics).to_csv(metrics_output_path, index=False)
    pd.concat(classification_reports).reset_index().to_csv(classification_report_output_path, index=False)
    pd.concat(confusion_matrices).to_csv(confusion_matrices_csv_path, index=True)

    # Compute RAGAS
    print("\n[✔] Computing RAGAS...")
    ragas_summary = []
    grouped = df.groupby(["embedding_model", "retrieval_method", "llm_model", "top_k", "temperature"])

    for name, group in tqdm(grouped, desc="RAGAS Evaluation"):
        records = []
        for _, row in group.iterrows():
            try:
                contexts = eval(row["contexts"]) if isinstance(row["contexts"], str) else row["contexts"]
            except Exception as e:
                contexts = []
                print(f"[⚠] Invalid context eval: {e}")

            records.append({
                "question": "Based on the patient data and TMN staging what is the exact stage of the cancer and the indicated course of treatment?",
                "contexts": contexts,
                "answer": row["raw_output"],
                "ground_truth": row["ground_truth_answer"]
            })

        dataset = Dataset.from_list(records)
        scores = defaultdict(list)

        for batch in chunk_dataset(dataset, chunk_size=100):
            ragas_result = compute_ragas_safe(batch)

            # Handle both dict and EvaluationResult formats safely
            if isinstance(ragas_result, dict):
                result_dict = ragas_result
            else:
                result_dict = {
                    "faithfulness": ragas_result["faithfulness"],
                    "answer_relevancy": ragas_result["answer_relevancy"],
                    "context_precision": ragas_result["context_precision"],
                    "context_recall": ragas_result["context_recall"],
                }

            for k, v in result_dict.items():
                scores[k].extend(v)

        ragas_summary.append({
            "embedding_model": name[0],
            "retrieval_method": name[1],
            "llm_model": name[2],
            "top_k": name[3],
            "temperature": name[4],
            "RAGAS_Faithfulness": np.nanmean(scores["faithfulness"]),
            "RAGAS_AnswerRelevancy": np.nanmean(scores["answer_relevancy"]),
            "RAGAS_ContextPrecision": np.nanmean(scores["context_precision"]),
            "RAGAS_ContextRecall": np.nanmean(scores["context_recall"]),
        })

    pd.DataFrame(ragas_summary).to_csv(ragas_output_path, index=False)
    print("[✔] Finished RAGAS evaluation safely.")