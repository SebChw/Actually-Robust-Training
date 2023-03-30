import torch
import torch.nn.functional as F


def measure_l1_for_source_separation(dataset, model, collate_fn):
    def measure_l1(examples):
        with torch.no_grad():
            # I need to change the format of it here from dict[str, list] to list[dict]
            # This is probably due to using torch sampler within data module and huggingface batching system inside of map function
            batch_of_examples = [{} for _ in range(3)]
            for key, value in examples.items():
                for i, v in enumerate(value):
                    batch_of_examples[i][key] = v

            batch = collate_fn(batch_of_examples)
            out = model.forward(batch["mixture"])
            loss = F.l1_loss(out, batch["target"], reduction="none").mean(dim=(2, 3))
            output = {
                f"{inst_name}L1": loss[:, i]
                for i, inst_name in enumerate(model.sources)
            }

            return output

    return dataset.map(
        measure_l1,
        fn_kwargs={"model": model, "collate_fn": collate_fn},
        batched=True,
        batch_size=3,
        drop_last_batch=True,
    )
