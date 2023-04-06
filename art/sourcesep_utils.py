import torch
import torch.nn.functional as F
from art.data.collate import create_sourceseparation_collate
import soundfile as sf


def reformat_examples(examples, batch_size):
    """Performs transformation dict[string, list] -> List[dict]"""
    batch_of_examples = [{} for _ in range(batch_size)]
    for key, value in examples.items():
        for i, v in enumerate(value):
            batch_of_examples[i][key] = v

    return batch_of_examples


def measure_l1_for_source_separation(
    dataset, model, collate_fn, load_from_cache=False, batch_size=3
):
    def measure_l1(examples):
        with torch.no_grad():
            # examples = examples.to_dict(orient="records") This doesn't work I wait for the answer on the forum.
            examples = reformat_examples(examples, batch_size)

            batch = collate_fn(examples)
            out = model.forward(batch["mixture"])
            loss = F.l1_loss(out, batch["target"], reduction="none").mean(dim=(2, 3))
            output = {
                f"{inst_name}L1": loss[:, i]
                for i, inst_name in enumerate(model.sources)
            }

            return output

    return dataset.map(
        measure_l1,
        batched=True,
        batch_size=batch_size,
        drop_last_batch=True,
        load_from_cache_file=load_from_cache,
        remove_columns=model.sources,
    )


def infer_and_save_results(dataset, model):
    collate_fn = create_sourceseparation_collate(instruments=model.sources)
    with torch.no_grad():
        for example in dataset:
            batch = collate_fn([example])
            out = model.forward(batch["mixture"]) * example["std"] + example["mean"]
            for i, instrument in enumerate(model.sources):
                sf.write(
                    f"{example['name'].split('/')[-1]}_window{example['n_window']}_{instrument}.wav",
                    out[0, i].cpu().numpy().T,
                    samplerate=44100,
                )
