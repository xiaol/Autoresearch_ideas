# Delivery-Time Prediction Brief

We want to predict the total delivery time for a local food order before the courier reaches the customer.

## Decision supported

- inform the customer of the expected arrival time
- identify likely late deliveries for intervention

## Prediction moment

At dispatch time, after food preparation finishes and the courier has been assigned.

## Available features

- route distance in kilometers
- kitchen preparation time
- courier load
- a simple weather severity score
- whether the order is in rush hour

## Constraints

- prediction should be cheap and fast
- baseline interpretability matters
- overpromising is costly because it damages trust
