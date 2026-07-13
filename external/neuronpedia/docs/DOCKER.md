# Docker Documentation

## Environment Variables and Configuration

The application uses a dotenv-based configuration system to manage environment variables across different deployment scenarios. When you build the Docker images, the entire contents of your chosen dotenv file (such as `.env.localhost`) are copied into the container through the `ENV_FILE` build argument. This approach ensures that all necessary configuration values are available within the container environment without having to manually specify each variable during the build process.

The reason for this approach is that it allows you to maintain different environment configurations for different deployment targets (localhost, staging, production) while keeping the Docker build process consistent. The startup scripts within the container will automatically load and apply these variables when the services initialize.

## Webapp Service

The webapp service is built from the Dockerfile located at `apps/webapp/Dockerfile`. This Dockerfile has been optimized to handle the complexities of building Node.js applications in network environments requiring custom CA bundles in the certificate chain (typically seen in corporate environments).

### The Standard Build vs Zero Trust Build

Modern enterprise environments often implement zero-trust security models that can significantly impact the Docker build process. The webapp Dockerfile is designed to gracefully handle two distinct scenarios:

#### Building in Standard Environments

If you're building on a personal machine or in an environment with standard internet access, the build process is straightforward. The system can reach external package repositories and download dependencies without additional configuration. In this case, simply running the make command will build all necessary components:

```bash
make webapp-localhost-build
```

This command orchestrates the entire build process, creating the webapp, database initialization service, and PostgreSQL containers in the correct sequence.

#### Building in Zero Trust Environments

Corporate environments often employ network security solutions like ZScaler that intercept and inspect all HTTPS traffic. While this provides important security benefits, it creates a challenge for Docker builds because the system attempts to verify SSL certificates against a different certificate authority than what the package managers expect.

To address this, the Dockerfile includes logic to handle custom Certificate Authority (CA) bundles. When you provide a custom CA bundle, the build process will:

1. Copy your CA bundle into the container's certificate store
2. Configure all relevant tools (npm, git, Node.js) to use this certificate chain
3. Ensure that SSL verification passes for all dependency downloads

To use this functionality, first copy your organization's CA bundle file to the project root directory, then build with the custom CA bundle specified:

```bash
make webapp-localhost-build CUSTOM_CA_BUNDLE=your-org-chain.txt
```

When the build runs successfully with a custom CA bundle, you'll see a confirmation message indicating which bundle file is being used.

It's worth noting that if you're in a zero-trust environment but don't have access to the appropriate CA bundle, you'll likely encounter SSL verification failures during the build process. The Dockerfile makes numerous HTTPS calls to download system packages and npm dependencies, and these will fail if the SSL certificate chain cannot be verified.

### Build Arguments and Their Purpose

The Dockerfile accepts two primary build arguments that control its behavior:

**CUSTOM_CA_BUNDLE**: This argument allows you to specify a custom certificate authority bundle file. When provided, the build process will integrate this bundle into the container's certificate store and configure all SSL-aware tools to use it. If not specified, it defaults to `.nocustomca`, which is a placeholder that prevents the CA bundle logic from executing.

**ENV_FILE**: This specifies which environment file should be copied into the container during the build process. This flexibility allows you to build containers for different environments (localhost, staging, production) from the same Dockerfile. The default value is `.env.localhost`, which is appropriate for local development.

### Runtime Environment Configuration

The webapp container sets several environment variables that are critical for proper SSL and network operation. These variables are automatically configured based on whether a custom CA bundle was provided during the build:

- `SSL_CERT_FILE` and `REQUESTS_CA_BUNDLE` ensure that HTTP libraries can verify SSL certificates
- `GIT_SSL_CAINFO` allows git operations to work with custom certificates
- `NODE_EXTRA_CA_CERTS` provides Node.js with additional certificate authorities
- Standard variables like `NODE_ENV`, `HOSTNAME`, and `PORT` control the application's runtime behavior

These variables are set automatically by the Dockerfile and generally shouldn't be modified unless you have specific requirements. The user-configurable variables that control your application's behavior are defined in the dotenv template files and are loaded by the startup scripts when the container initializes.

## Database Initialization Strategy

The database initialization process uses an interesting architectural pattern that's worth understanding. Rather than having the webapp service handle database setup directly, there's a dedicated `db-init` service that manages all database preparation tasks. This service shares the same Dockerfile as the webapp but uses a different build target, which allows it to include additional tools and dependencies that are only needed for database operations.

### Multi-Stage Build Architecture

The Dockerfile employs a multi-stage build strategy that creates different images for different purposes:

The `db-init` stage includes development dependencies like TypeScript and ts-node, which are necessary for running database migration scripts and seed data operations. This stage has access to the full development toolchain because database initialization often requires running complex TypeScript files.

The `runner` stage, used by the webapp service, contains only the production dependencies and the built application. This creates a much smaller, more secure container for the actual web application since it doesn't include development tools that could pose security risks in production.

### Service Dependency Chain

The services are orchestrated with a specific dependency chain that ensures proper initialization order:

```
PostgreSQL Database → Database Init Service → Webapp Service
```

This means that the PostgreSQL container must be healthy before the database initialization service starts, and the database initialization must complete successfully before the webapp service begins. This dependency chain prevents race conditions and ensures that the database is fully prepared before the application attempts to connect to it.

The database initialization service runs once and then exits (`restart: "no"`), while the webapp service continues running to serve requests. This pattern ensures that database setup operations don't interfere with the running application.

## Docker Compose Architecture

The `compose.yaml` file defines a complete application stack with five interconnected services, each serving a specific purpose in the overall system:

**webapp**: The main Next.js application that serves the user interface and handles web requests on port 3000.

**db-init**: A one-time service that prepares the database by running migrations and seeding initial data.

**postgres**: A PostgreSQL database with the pgvector extension, which provides vector similarity search capabilities. The database includes a health check to ensure it's ready before dependent services start.

**inference**: A machine learning inference service that handles model predictions and analysis on port 5002.

**autointerp**: An auto-interpretation service that provides automated analysis capabilities on port 5003.

### Volume and Network Configuration

The compose configuration includes persistent storage for the PostgreSQL database through the `postgres_data` volume. This ensures that your data survives container restarts and updates.

The database initialization scripts are mounted from the host filesystem, allowing you to modify database setup procedures without rebuilding the container. This is particularly useful during development when you might need to adjust database schemas or seed data.

All services communicate through a dedicated Docker network (`neuronpedia-network`), which provides isolated networking and allows services to communicate using service names as hostnames.

## Make Commands and Workflow

The project includes Make commands that simplify common Docker operations while handling the complexity of environment configuration:

### Building the Application

The build command orchestrates the creation of all necessary Docker images:

```bash
make webapp-localhost-build [CUSTOM_CA_BUNDLE=<filename>]
```

This command first checks that Docker is installed on your system, then proceeds to build the webapp, database initialization, and PostgreSQL images. The build process respects the `CUSTOM_CA_BUNDLE` parameter if provided, ensuring that corporate environments can build successfully.

### Running the Application

The run command brings up the entire application stack:

```bash
make webapp-localhost-run
```

This command starts all services in the correct order, respecting the dependency chain and ensuring that each service has the environment configuration it needs to operate properly.

Both commands include error checking to provide helpful feedback if Docker isn't available or if other prerequisites aren't met. This makes the development experience more pleasant by catching common issues early and providing clear guidance on how to resolve them.