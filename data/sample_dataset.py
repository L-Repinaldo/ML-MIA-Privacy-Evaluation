def sample_dataset_bundle(
    dataset_bundle,
    sample_size,
    random_state,
):
    sampled = []

    for df in dataset_bundle["datasets"]:

        if len(df) > sample_size:

            df = df.sample(
                n=sample_size,
                random_state=random_state,
            )

        sampled.append(df.reset_index(drop=True))

    dataset_bundle["datasets"] = sampled

    return dataset_bundle