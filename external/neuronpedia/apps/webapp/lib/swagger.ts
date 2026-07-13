import { createSwaggerSpec } from 'next-swagger-doc';

const description = `
This is Neuronpedia's API. You can test calls right in the browser.

IMPORTANT: Check the [available resources](https://neuronpedia.org/available-resources) page to see which models and sources are available in the API.

API KEY: Get your API key by logging in at [neuronpedia.org](https://www.neuronpedia.org) then going to [/account](https://neuronpedia.org/account).

Requests for new APIs, questions, or feedback? [Contact us](https://neuronpedia.org/contact).
`;

export const getApiDocs = async () => {
  // changes here should also be added to next-swagger-doc.json
  const spec = createSwaggerSpec({
    apiFolder: 'app/api',
    definition: {
      openapi: '3.0.0',
      authentication: {
        preferredSecurityScheme: 'apiKey',
      },
      info: {
        title: 'Neuronpedia API',
        description,
        version: '1.0',
      },
      components: {
        securitySchemes: {
          apiKey: {
            type: 'apiKey',
            in: 'header',
            name: 'x-api-key',
          },
        },
        schemas: {
          Explanation: {
            type: 'object',
            description: 'Schema for a returned explanation or explanations.',
            properties: {
              modelId: {
                type: 'string',
              },
              layer: {
                type: 'string',
              },
              index: {
                type: 'string',
              },
              description: {
                type: 'string',
              },
              typeName: {
                type: 'string',
              },
              explanationModelName: {
                type: 'string',
              },
            },
          },
        },
      },
      security: [],
    },
  });
  return spec;
};
