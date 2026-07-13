# ActivationTopkByTokenBatchPost200Response

Response for NPActivationTopkByTokenBatchRequest. Contains the batch results of top features at each token position and the tokenized prompts.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**results** | [**List[ActivationTopkByTokenBatchPost200ResponseResultsInner]**](ActivationTopkByTokenBatchPost200ResponseResultsInner.md) |  | 

## Example

```python
from neuronpedia_inference_client.models.activation_topk_by_token_batch_post200_response import ActivationTopkByTokenBatchPost200Response

# TODO update the JSON string below
json = "{}"
# create an instance of ActivationTopkByTokenBatchPost200Response from a JSON string
activation_topk_by_token_batch_post200_response_instance = ActivationTopkByTokenBatchPost200Response.from_json(json)
# print the JSON string representation of the object
print(ActivationTopkByTokenBatchPost200Response.to_json())

# convert the object into a dict
activation_topk_by_token_batch_post200_response_dict = activation_topk_by_token_batch_post200_response_instance.to_dict()
# create an instance of ActivationTopkByTokenBatchPost200Response from a dict
activation_topk_by_token_batch_post200_response_from_dict = ActivationTopkByTokenBatchPost200Response.from_dict(activation_topk_by_token_batch_post200_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


