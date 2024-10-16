from tqdm import tqdm
import numpy as np
import torch
import clip


def compute_retrieval(a2b_sims, return_ranks=False):
    """
    Args:
        a2b_sims: Result of computing similarity between two sets of embeddings (emb1 @ emb2.T)
            with shape (num_datapoints, num_datapoints).

    Returns:
        Retrieval metrics for that similarity.
    """
    npts = a2b_sims.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    # loop source embedding indices
    for index in range(npts):
        # get order of similarities to target embeddings
        inds = np.argsort(a2b_sims[index])[::-1]
        # find where the correct embedding is ranked
        where = np.where(inds == index)
        rank = where[0][0]
        ranks[index] = rank
        # save the top1 result as well
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    report_dict = {"R@1": r1, "R@5": r5, "R@10": r10} 

    if return_ranks:
        return report_dict, (ranks, top1)
    else:
        return report_dict


@torch.no_grad()
def coco_captions_eval(model, loader, progress_bar=True, device="cuda", using_clip=False):
    correct, total = 0, 0

    if progress_bar:
        bar = tqdm(total=len(loader))
    logit_scale = torch.tensor(np.log(100.0)).to(device)

    image_store = []
    text_store = []
    D_img = 0

    for idx, (images, captions) in enumerate(loader):
        batch_size = images.shape[0]
        images = images.float().to(device)
        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        D_img = image_features.shape[-1]
        # image_features = image_features.repeat((5, 1))
        image_store.append(image_features)

        # average the embeddings of the 5 captions per image
        caption_store = []
        for item in captions:
            item = item[:5]  #[item[0]]
            if using_clip:
                item = clip.tokenize(item).to(device)
            item_embeddings = model.encode_text(item)
            item_embeddings /= item_embeddings.norm(dim=-1, keepdim=True)
            caption_store.append(item_embeddings.unsqueeze(0))
        text_store.append(torch.cat(caption_store, dim=0))
        
        if progress_bar:
            bar.set_postfix({"batch_index": idx})
            bar.update(1)
        
    image_store = torch.cat(image_store, dim=0).view(5000, D_img)
    text_store = torch.cat(text_store, dim=0).view(5000, 5, D_img)

    mean_recalls = {"R@1": 0, "R@5": 0, "R@10": 0}
    for i in range(1):
        sim = logit_scale * (image_store @ text_store[:, i, :].T)
        res = compute_retrieval(sim.cpu().numpy())
        for k in res.keys():
            mean_recalls[k] += res[k]
    
    return mean_recalls


@torch.no_grad()
def image_classification_eval(model, loader, progress_bar=True, device="cuda", using_clip=False):
    correct, total = 0, 0

    if progress_bar:
        bar = tqdm(total=len(loader))

    logit_scale = torch.tensor(np.log(100.0)).to(device)
    class_prompt = [f"a photo of a {c}" for c in loader.dataset.classes]
    
    if using_clip:
        class_prompt = clip.tokenize(class_prompt)
    class_features = model.encode_text(class_prompt).to(device)

    for idx, (images, labels) in enumerate(loader):
        batch_size = images.shape[0]

        if using_clip:
            captions = clip.tokenize(captions).to(device)

        images = images.float().to(device)
        image_features = model.encode_image(images)

        sim = logit_scale.exp() * image_features @ class_features.T
        preds = torch.argmax(sim, dim=-1)
        correct += (preds == labels).sum().item()
        total += batch_size

        accuracy = round(correct/total * 100, 2)

        if progress_bar:
            bar.set_postfix({"accuracy": accuracy})

        bar.update(1)
    
    return accuracy