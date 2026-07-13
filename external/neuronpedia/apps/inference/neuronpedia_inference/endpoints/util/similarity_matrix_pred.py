import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch

from neuronpedia_inference.sae_manager import SAEManager
from neuronpedia_inference.shared import (
    with_request_lock,
    Model,
)

logger = logging.getLogger(__name__)

router = APIRouter()


class SimilarityMatrixPostRequest(BaseModel):
    modelId: str
    sourceId: str
    index: int
    text: str


@router.post("/util/similarity-matrix-pred")
@with_request_lock()
async def similarity_matrix(request: SimilarityMatrixPostRequest):
    model = Model.get_instance()
    source = request.sourceId
    index = request.index

    sae = SAEManager.get_instance().get_sae(source)

    # if the sae architecture is not temporal, throw error
    if sae.cfg.architecture() != "temporal":
        logger.error("SAE architecture is not temporal")
        return JSONResponse(
            content={"error": "SAE architecture is not temporal"},
            status_code=400,
        )

    # tokenize the text
    prepend_bos = model.cfg.tokenizer_prepends_bos
    tokens = model.to_tokens(
        request.text,
        prepend_bos=prepend_bos,
        truncate=False,
    )[0]
    _, cache = model.run_with_cache(tokens)
    logger.info("tokens: %s", tokens)

    str_tokens = model.to_str_tokens(request.text, prepend_bos=prepend_bos)
    logger.info("str_tokens: %s", str_tokens)

    hook_name = sae.cfg.metadata.hook_name
    _, z_pred = sae.encode_with_predictions(cache[hook_name])

    # Extract single batch sample
    pred_LD = z_pred[0]  # Shape: [L, D]
    
    # Center the predictions
    pred_centered_LD = pred_LD - torch.mean(pred_LD, dim=0, keepdim=True)
    
    # Normalize along the D dimension
    pred_LD_normalized = torch.nn.functional.normalize(pred_centered_LD.float(), p=2, dim=-1)  # L x D, normalized along D
    
    # Compute cosine similarity: pred_LD @ pred_LD.T -> L x L
    cosine_sim_LL = pred_LD_normalized @ pred_LD_normalized.T
    cosine_sim_np = cosine_sim_LL.detach().cpu().numpy()

    # remove the bos token
    str_tokens = str_tokens[1:]
    cosine_sim_np = cosine_sim_np[1:, 1:]
    
    return JSONResponse(
        content={"similarity_matrix": cosine_sim_np.tolist(), "tokens": str_tokens},
        status_code=200,
    )
