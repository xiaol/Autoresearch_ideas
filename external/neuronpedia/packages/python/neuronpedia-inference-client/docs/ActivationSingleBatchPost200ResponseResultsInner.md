# ActivationSingleBatchPost200ResponseResultsInner


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**activation** | [**ActivationSinglePost200ResponseActivation**](ActivationSinglePost200ResponseActivation.md) |  | 
**tokens** | **List[str]** |  | 

## Example

```python
from neuronpedia_inference_client.models.activation_single_batch_post200_response_results_inner import ActivationSingleBatchPost200ResponseResultsInner

# TODO update the JSON string below
json = "{}"
# create an instance of ActivationSingleBatchPost200ResponseResultsInner from a JSON string
activation_single_batch_post200_response_results_inner_instance = ActivationSingleBatchPost200ResponseResultsInner.from_json(json)
# print the JSON string representation of the object
print(ActivationSingleBatchPost200ResponseResultsInner.to_json())

# convert the object into a dict
activation_single_batch_post200_response_results_inner_dict = activation_single_batch_post200_response_results_inner_instance.to_dict()
# create an instance of ActivationSingleBatchPost200ResponseResultsInner from a dict
activation_single_batch_post200_response_results_inner_from_dict = ActivationSingleBatchPost200ResponseResultsInner.from_dict(activation_single_batch_post200_response_results_inner_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


