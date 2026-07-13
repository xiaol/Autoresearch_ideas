# ActivationTopkByTokenBatchPost200ResponseResultsInner


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**results** | [**List[ActivationTopkByTokenPost200ResponseResultsInner]**](ActivationTopkByTokenPost200ResponseResultsInner.md) |  | 
**tokens** | **List[str]** |  | 

## Example

```python
from neuronpedia_inference_client.models.activation_topk_by_token_batch_post200_response_results_inner import ActivationTopkByTokenBatchPost200ResponseResultsInner

# TODO update the JSON string below
json = "{}"
# create an instance of ActivationTopkByTokenBatchPost200ResponseResultsInner from a JSON string
activation_topk_by_token_batch_post200_response_results_inner_instance = ActivationTopkByTokenBatchPost200ResponseResultsInner.from_json(json)
# print the JSON string representation of the object
print(ActivationTopkByTokenBatchPost200ResponseResultsInner.to_json())

# convert the object into a dict
activation_topk_by_token_batch_post200_response_results_inner_dict = activation_topk_by_token_batch_post200_response_results_inner_instance.to_dict()
# create an instance of ActivationTopkByTokenBatchPost200ResponseResultsInner from a dict
activation_topk_by_token_batch_post200_response_results_inner_from_dict = ActivationTopkByTokenBatchPost200ResponseResultsInner.from_dict(activation_topk_by_token_batch_post200_response_results_inner_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


