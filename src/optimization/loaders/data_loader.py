def create_dataloader(dataset, batch_size, num_workers=0, worker_init_fn=seed_worker):
    generator = torch.Generator()
    generator.manual_seed(42)  # Seed the generator
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
        worker_init_fn=seed_worker
    )