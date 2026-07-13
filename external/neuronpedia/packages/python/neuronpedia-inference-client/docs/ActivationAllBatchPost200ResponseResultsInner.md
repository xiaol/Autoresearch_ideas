# ActivationAllBatchPost200ResponseResultsInner


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**activations** | [**List[ActivationAllPost200ResponseActivationsInner]**](ActivationAllPost200ResponseActivationsInner.md) |  | 
**tokens** | **List[str]** |  | 
**counts** | **List[List[float]]** | Not currently supported and may be incorrect. This is the number of features that activated by layer, starting from layer 0 of this SAE. Need to be redesigned. | [optional] 

## Example

```python
from neuronpedia_inference_client.models.activation_all_batch_post200_response_results_inner import ActivationAllBatchPost200ResponseResultsInner

# TODO update the JSON string below
json = "{}"
# create an instance of ActivationAllBatchPost200ResponseResultsInner from a JSON string
activation_all_batch_post200_response_results_inner_instance = ActivationAllBatchPost200ResponseResultsInner.from_json(json)
# print the JSON string representation of the object
print(ActivationAllBatchPost200ResponseResultsInner.to_json())

# convert the object into a dict
activation_all_batch_post200_response_results_inner_dict = activation_all_batch_post200_response_results_inner_instance.to_dict()
# create an instance of ActivationAllBatchPost200ResponseResultsInner from a dict
activation_all_batch_post200_response_results_inner_from_dict = ActivationAllBatchPost200ResponseResultsInner.from_dict(activation_all_batch_post200_response_results_inner_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


