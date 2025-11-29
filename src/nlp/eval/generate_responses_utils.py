import pandas as pd
import re
import time
from collections import defaultdict
from tqdm import tqdm
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score
from sklearn.utils.multiclass import unique_labels
from datasets import Dataset
import numpy as np
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas import evaluate
from bert_score import score as bert_score


# Carregar datasets
def load_ground_truth():
    df = pd.read_csv(GROUND_TRUTH_PATH, sep=";", engine="python")
    df.columns = ["Cancer Type", "T-Stage", "N-Stage", "M-Stage", "Stage", "Age", "Treatment Plan"]
    df.dropna(subset=["T-Stage", "N-Stage", "M-Stage", "Stage", "Treatment Plan"], inplace=True)
    return df

def load_patient_inputs():
    df = pd.read_csv(INPUT_PATIENTS_PATH, sep=';')
    expected_cols = {"T_stage", "N_stage", "M_stage", "age", "Gender", "smoking_status", "cancer_type"}
    if not expected_cols.issubset(set(df.columns)):
        raise ValueError(f"Missing columns in input file. Required: {expected_cols}")
    return df

# Verificação do matching
def test_stage_matching():
    gt = load_ground_truth()
    patients = load_patient_inputs()

    matched = []
    unmatched = []

    print("\n========= Patient Stage Matching =========\n")

    for idx, row in patients.iterrows():
        t = str(row["T_stage"]).strip()
        n = str(row["N_stage"]).strip()
        m = str(row["M_stage"]).strip()
        cancer_type = str(row["cancer_type"])
        age = int(row["age"])

        # SCLC: Matching com múltiplos valores por linha e faixa etária
        if "SMALL CELL" in cancer_type.upper():
            match_found = False
            for _, ref in gt[gt["Cancer Type"].str.upper().str.contains("SMALL CELL")].iterrows():
                t_vals = [v.strip() for v in str(ref["T-Stage"]).split(";")]
                n_vals = [v.strip() for v in str(ref["N-Stage"]).split(";")]
                m_vals = [v.strip() for v in str(ref["M-Stage"]).split(";")]
                age_label = str(ref["Age"]).strip()

                t_match = "Any T" in t_vals or t in t_vals
                n_match = "Any N" in n_vals or n in n_vals
                m_match = "Any M" in m_vals or m in m_vals or any(m == mv for mv in m_vals)
                age_group = ">= 70" if age >= 70 else "< 70"

                if t_match and n_match and m_match and age_label == age_group:
                    stage = ref["Stage"]
                    matched.append((idx, stage))
                    print(f"[✔] Patient {idx}: TNM: {t}-{n}-{m} | Age: {age} | Type: {cancer_type} → Matched Stage: **{stage}**")
                    match_found = True
                    break

            if not match_found:
                unmatched.append((idx, t, n, m, cancer_type))
                print(f"[✖] Patient {idx}: No SCLC match for TNM: {t}-{n}-{m} | Age: {age} | Type: {cancer_type}")

        # NSCLC: matching direto ou por regras de M1
        else:
            if m in ["M1a", "M1b", "M1c"]:
                match = gt[
                    (gt["M-Stage"] == m) &
                    (gt["T-Stage"].str.contains("Any", case=False)) &
                    (gt["N-Stage"].str.contains("Any", case=False))
                ]
            elif m == "M1":
                match = gt[
                    (gt["M-Stage"].str.startswith("M1")) &
                    (gt["T-Stage"].str.contains("Any", case=False)) &
                    (gt["N-Stage"].str.contains("Any", case=False))
                ]
            else:
                match = gt[
                    (gt["T-Stage"] == t) &
                    (gt["N-Stage"] == n) &
                    (gt["M-Stage"] == m)
                ]

            if match.empty:
                unmatched.append((idx, t, n, m, cancer_type))
                print(f"[✖] Patient {idx}: No NSCLC match for TNM: {t}-{n}-{m} | Type: {cancer_type}")
            else:
                stage = match.iloc[0]["Stage"]
                matched.append((idx, stage))
                print(f"[✔] Patient {idx}: TNM: {t}-{n}-{m} | Type: {cancer_type} → Matched Stage: **{stage}**")

    print(f"\nResumo: {len(matched)} pacientes com estágio atribuído | {len(unmatched)} sem correspondência\n")

    return matched, unmatched

#  STAGE & TREATMENT EXTRACTION
def extract_stage_and_treatment(generated_output):
    text = generated_output.upper()
    predicted_stage = None

    # SCLC classification (robust version)
    if re.search(r"\b(EXTENSIVE[\s\-]?STAGE|ES\-SCLC)\b", text):
        predicted_stage = "ES-SCLC"
    elif re.search(r"\b(LIMITED[\s\-]?STAGE|LS\-SCLC)\b", text):
        predicted_stage = "LS-SCLC"
    else:
        # NSCLC staging extraction: Look for full Stage declaration (e.g., "Stage IA")
        stage_matches = re.findall(r"\bSTAGE\s+(I{1,3}|IV)([ABC])?\b", text)
        if stage_matches:
            # Build full stage string (e.g., 'IA', 'IIIB', etc.)
            stages = [f"{m[0]}{m[1] or ''}" for m in stage_matches]
            predicted_stage = stages[-1]  # Take the last matched stage
        else:
            # Fallback: Check for exact TNM grouping (e.g., "T1a, N0, M0") and infer known combinations
            tnms = re.search(r"\bT\d[AaBb]?,?\s*N\d[AaBb]?,?\s*M\d[AaBb]?\b", text)
            if tnms:
                # Optional: integrate logic to map TNM to stage using AJCC rules, if needed
                predicted_stage = None  # placeholder if future logic needed

    # Treatment extraction (same as before, robust)
    treatment_match = re.split(r"(?i)treatment plan:|appropriate treatment plan is:", generated_output)
    predicted_treatment = treatment_match[1].strip() if len(treatment_match) > 1 else generated_output.strip()

    return predicted_stage, predicted_treatment


def enforce_rate_limits(token_count):
    """Dynamically enforce OpenAI rate limits before making a request."""
    global REQUEST_TIMESTAMPS, TOKENS_USED
    current_time = time.time()

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


#  MAIN FUNCTION  #
def run_evaluation_pipeline():
    test_set = load_ground_truth()
    patient_inputs = load_patient_inputs()
    all_results = []
    ragas_records_map = defaultdict(list)

    for idx, patient in tqdm(patient_inputs.iterrows(), total=len(patient_inputs)):
        t_stage = patient["T_stage"]
        n_stage = patient["N_stage"]
        m_stage = patient["M_stage"]
        age = int(patient["age"])
        cancer_type = patient["cancer_type"]

        if "SMALL CELL" in cancer_type.upper():
            # Matching para SCLC
            stage_found = False
            for _, row in test_set[test_set["Cancer Type"].str.upper().str.contains("SMALL CELL")].iterrows():
                t_vals = [v.strip() for v in row["T-Stage"].split(";")]
                n_vals = [v.strip() for v in row["N-Stage"].split(";")]
                m_vals = [v.strip() for v in row["M-Stage"].split(";")]
                age_label = row["Age"].strip()

                t_match = "Any T" in t_vals or t_stage in t_vals
                n_match = "Any N" in n_vals or n_stage in n_vals
                m_match = "Any M" in m_vals or m_stage in m_vals or any(m_stage == mv for mv in m_vals)

                age_group = ">= 70" if age >= 70 else "< 70"

                if t_match and n_match and m_match and age_group == age_label:
                    true_stage = row["Stage"]
                    true_treatment = row["Treatment Plan"]
                    stage_found = True
                    break

            if not stage_found:
                print(f"[Warning] No SCLC match for TNM: {t_stage}, {n_stage}, {m_stage} and age: {age}")
                continue

        else:
            # Matching para NSCLC
            match = test_set[
                (test_set["T-Stage"] == t_stage) &
                (test_set["N-Stage"] == n_stage) &
                (test_set["M-Stage"] == m_stage)
            ]

            # Se não encontrar, tenta fallback com "Any"
            if match.empty and m_stage in ["M1a", "M1b", "M1c"]:
                match = test_set[
                    (test_set["M-Stage"] == m_stage) &
                    ((test_set["T-Stage"] == "Any T") | test_set["T-Stage"].str.contains("Any", case=False)) &
                    ((test_set["N-Stage"] == "Any N") | test_set["N-Stage"].str.contains("Any", case=False))
                ]
            elif match.empty and m_stage == "M1":
                match = test_set[
                    (test_set["M-Stage"].str.startswith("M1")) &
                    ((test_set["T-Stage"] == "Any T") | test_set["T-Stage"].str.contains("Any", case=False)) &
                    ((test_set["N-Stage"] == "Any N") | test_set["N-Stage"].str.contains("Any", case=False))
                ]

            if match.empty:
                print(f"[Warning] No NSCLC match for TNM: {t_stage}, {n_stage}, {m_stage}")
                continue

            true_stage = match.iloc[0]["Stage"]
            true_treatment = match.iloc[0]["Treatment Plan"]

        age = int(patient["age"])
        cancer_type = patient["cancer_type"]
        smoker = patient["smoking_status"]
        gender = patient["Gender"]
        additional_info = f"Smoker: {'Yes' if smoker else 'No'}"
        
        for embedding_model in EMBEDDING_MODELS:
            for retrieval_method in RETRIEVAL_METHODS:
                for llm_model in LLM_MODELS:
                    for run_idx in range(N_REPEATS):
                        max_retries = 1
                        retry_delay = 15  # segundos

                        success = False
                        for attempt in range(max_retries):
                            try:
                                enforce_rate_limits(6000)
                                response, retrieved_docs = retrieval_and_response_pipeline(
                                    query="Based on the patient data and TMN staging what is the exact stage of the cancer and the indicated course of treatment?",
                                    embedding_model=embedding_model,
                                    retrieval_method=retrieval_method,
                                    llm_model=llm_model,
                                    t_stage=t_stage,
                                    n_stage=n_stage,
                                    m_stage=m_stage,
                                    histopath_grade="",
                                    cancer_type=cancer_type,
                                    age=age,
                                    gender=gender,
                                    additional_info=additional_info
                                )

                                if response and not str(response).startswith("ERROR"):
                                    success = True
                                    break  # sucesso, sair do loop
                            except Exception as e:
                                print(f"[Attempt {attempt+1}/{max_retries}] API Error: {e}")

                            print(f"[Attempt {attempt+1}/{max_retries}] Retrying in {retry_delay} seconds...")
                            time.sleep(retry_delay)

                        if not success:
                            print(f"Failed after {max_retries} retries. Skipping input_idx={idx}.")
                            continue


                        pred_stage, pred_treatment = extract_stage_and_treatment(response)

                        all_results.append({
                            "input_idx": idx,
                            "run_idx": run_idx,
                            "embedding_model": embedding_model,
                            "retrieval_method": retrieval_method,
                            "llm_model": llm_model,
                            "T_stage": t_stage,
                            "N_stage": n_stage,
                            "M_stage": m_stage,
                            "cancer_type": cancer_type,
                            "true_stage": true_stage,
                            "predicted_stage": pred_stage,
                            "ground_truth_answer": true_treatment,
                            "predicted_treatment": pred_treatment,
                            "raw_output": response,
                            "contexts": retrieved_docs
                        })

                        combo_key = (embedding_model, retrieval_method, llm_model)
                        ragas_records_map[combo_key].append({
                            "question": "Based on the patient data and TMN staging what is the exact stage of the cancer and the indicated course of treatment?",
                            "answer": response,
                            "contexts": retrieved_docs,
                            "ground_truths": [true_treatment],
                            "reference": true_treatment
                        })

    results_df = pd.DataFrame(all_results)
    results_df["stage_match"] = results_df["true_stage"] == results_df["predicted_stage"]
    
    results_df.to_csv(OUTPUT_PATH, index=False)
    print(f"[✔] Saved results to {OUTPUT_PATH}")

    print("Computing RAGAS...")
    ragas_summary = []
    for combo_key, records in ragas_records_map.items():
        embedding_model, retrieval_method, llm_model = combo_key
        ragas_dataset = Dataset.from_list(records)


        ragas_result = evaluate(
            dataset=ragas_dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=llm,
            embeddings=embeddings
        )
        
        ragas_summary.append({
            "embedding_model": embedding_model,
            "retrieval_method": retrieval_method,
            "llm_model": llm_model,
            "RAGAS_Faithfulness": np.nanmean(ragas_result["faithfulness"]),
            "RAGAS_AnswerRelevancy": np.nanmean(ragas_result["answer_relevancy"]),
            "RAGAS_ContextPrecision": np.nanmean(ragas_result["context_precision"]),
            "RAGAS_ContextRecall": np.nanmean(ragas_result["context_recall"])
        })

        mask = (
            (results_df["embedding_model"] == embedding_model) &
            (results_df["retrieval_method"] == retrieval_method) &
            (results_df["llm_model"] == llm_model)
        )

        for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            mean_value = np.nanmean(ragas_result[metric])
            results_df.loc[mask, f"RAGAS_{metric}"] = mean_value


    print("Computing BERTScore...")
    preds = results_df["predicted_treatment"].fillna("").tolist()
    refs = results_df["ground_truth_answer"].fillna("").tolist()
    _, _, f1_scores = bert_score(preds, refs, lang="en", verbose=False, device="mps")
    results_df["bertscore_f1"] = f1_scores.numpy()

    print("Computing BLEU and ROUGE-L...")
    bleu_scores, rouge_l_scores = [], []
    smoother = SmoothingFunction()
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    for pred, ref in zip(preds, refs):
        bleu = sentence_bleu([ref.split()], pred.split(), smoothing_function=smoother.method1)
        rouge_l = rouge.score(ref, pred)["rougeL"].fmeasure
        bleu_scores.append(bleu)
        rouge_l_scores.append(rouge_l)
    results_df["bleu"] = bleu_scores
    results_df["rougeL"] = rouge_l_scores


    results_df.to_csv(OUTPUT_PATH, index=False)
    print(f"[✔] Saved results to {OUTPUT_PATH}")
    
    print("Computing Classification Metrics for Staging...")
    valid_results = results_df.dropna(subset=["true_stage", "predicted_stage"])
    labels = unique_labels(valid_results["true_stage"], valid_results["predicted_stage"])

    # Matriz de confusão com labels fixos
    cm = confusion_matrix(valid_results["true_stage"], valid_results["predicted_stage"], labels=labels)

    # Construção segura do DataFrame
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv("/Users/catarinasilva/Desktop/LLM/confusion_matrix.csv")
    print(f"[✔] Saved confusion matrix to confusion_matrix.csv")

    cls_report = classification_report(valid_results["true_stage"], valid_results["predicted_stage"], output_dict=True, zero_division=0)
    cls_report_df = pd.DataFrame(cls_report).transpose()
    cls_report_df.to_csv("/Users/catarinasilva/Desktop/LLM/classification_report.csv")
    print(f"[✔] Saved classification report to classification_report.csv")

    stage_accuracy = accuracy_score(valid_results["true_stage"], valid_results["predicted_stage"])
    stage_f1_macro = f1_score(valid_results["true_stage"], valid_results["predicted_stage"], average='macro', zero_division=0)
    stage_f1_weighted = f1_score(valid_results["true_stage"], valid_results["predicted_stage"], average='weighted', zero_division=0)
    stage_precision_macro = precision_score(valid_results["true_stage"], valid_results["predicted_stage"], average='macro', zero_division=0)
    stage_recall_macro = recall_score(valid_results["true_stage"], valid_results["predicted_stage"], average='macro', zero_division=0)
    stage_kappa = cohen_kappa_score(valid_results["true_stage"], valid_results["predicted_stage"])

    summary = {
        "Stage Accuracy": stage_accuracy,
        "Stage F1 Score (Macro)": stage_f1_macro,
        "Stage F1 Score (Weighted)": stage_f1_weighted,
        "Stage Precision (Macro)": stage_precision_macro,
        "Stage Recall (Macro)": stage_recall_macro,
        "Stage Cohen's Kappa": stage_kappa,
        "BERTScore F1 (mean)": results_df["bertscore_f1"].mean(),
        "BLEU Score (mean)": results_df["bleu"].mean(),
        "ROUGE-L Score (mean)": results_df["rougeL"].mean(),
        "RAGAS_Faithfulness": np.nanmean(ragas_result["faithfulness"]),
        "RAGAS_AnswerRelevancy": np.nanmean(ragas_result["answer_relevancy"]),
        "RAGAS_ContextPrecision": np.nanmean(ragas_result["context_precision"]),
        "RAGAS_ContextRecall": np.nanmean(ragas_result["context_recall"])
    }

    pd.DataFrame([summary]).to_csv(SUMMARY_PATH, index=False)
    print(f"[✔] Saved summary to {SUMMARY_PATH}")

    grouped_df = results_df.groupby(["embedding_model", "retrieval_method", "llm_model"])[
        ["stage_match", "bertscore_f1", "bleu", "rougeL"]
    ].agg(["mean", "std"]).reset_index()
    
    grouped_df.columns = ['embedding_model', 'retrieval_method', 'llm_model'] + ['_'.join(col).strip() for col in grouped_df.columns.values[3:]]

    # Carrega as métricas RAGAS por combinação
    ragas_summary_df = pd.DataFrame(ragas_summary)

    # Junta as métricas por chave
    full_grouped = pd.merge(grouped_df, ragas_summary_df, on=["embedding_model", "retrieval_method", "llm_model"], how="left")

    # Salva tudo junto
    full_grouped.to_csv(GROUPED_PATH, index=False)
    print(f"[✔] Saved grouped metrics including RAGAS to {GROUPED_PATH}")