# ActivationAllBatchPost200Response

Response for NPActivationAllBatchRequest. Contains the batch results of activations for each top feature and the tokenized prompts.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**results** | [**List[ActivationAllBatchPost200ResponseResultsInner]**](ActivationAllBatchPost200ResponseResultsInner.md) |  | 

## Example

```python
from neuronpedia_inference_client.models.activation_all_batch_post200_response import ActivationAllBatchPost200Response

# TODO update the JSON string below
json = "{}"
# create an instance of ActivationAllBatchPost200Response from a JSON string
activation_all_batch_post200_response_instance = ActivationAllBatchPost200Response.from_json(json)
# print the JSON string representation of the object
print(ActivationAllBatchPost200Response.to_json())

# convert the object into a dict
activation_all_batch_post200_response_dict = activation_all_batch_post200_response_instance.to_dict()
# create an instance of ActivationAllBatchPost200Response from a dict
activation_all_batch_post200_response_from_dict = ActivationAllBatchPost200Response.from_dict(activation_all_batch_post200_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


