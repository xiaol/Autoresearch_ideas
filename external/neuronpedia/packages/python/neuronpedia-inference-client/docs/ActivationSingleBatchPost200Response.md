# ActivationSingleBatchPost200Response

Response for NPActivationBatchRequest. Contains the batch results of activation values and tokenized prompt.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**results** | [**List[ActivationSingleBatchPost200ResponseResultsInner]**](ActivationSingleBatchPost200ResponseResultsInner.md) |  | 

## Example

```python
from neuronpedia_inference_client.models.activation_single_batch_post200_response import ActivationSingleBatchPost200Response

# TODO update the JSON string below
json = "{}"
# create an instance of ActivationSingleBatchPost200Response from a JSON string
activation_single_batch_post200_response_instance = ActivationSingleBatchPost200Response.from_json(json)
# print the JSON string representation of the object
print(ActivationSingleBatchPost200Response.to_json())

# convert the object into a dict
activation_single_batch_post200_response_dict = activation_single_batch_post200_response_instance.to_dict()
# create an instance of ActivationSingleBatchPost200Response from a dict
activation_single_batch_post200_response_from_dict = ActivationSingleBatchPost200Response.from_dict(activation_single_batch_post200_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


