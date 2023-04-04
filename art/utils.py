import torch
import torch.nn.functional as F


def measure_l1_for_source_separation(dataset, model, collate_fn, load_from_cache=False):
    def measure_l1(examples):
        with torch.no_grad():
            examples = examples.to_dict(orient="records")

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
        fn_kwargs={"model": model, "collate_fn": collate_fn},
        batched=True,
        batch_size=3,
        drop_last_batch=True,
        load_from_cache_file=load_from_cache,
    )
