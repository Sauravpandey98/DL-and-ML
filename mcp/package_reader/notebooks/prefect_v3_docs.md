Get started
# Introduction
Prefect is an open-source orchestration engine that turns your Python functions into production-grade data pipelines with minimal friction. You can build and schedule workflows in pure Python—no DSLs or complex config files—and run them anywhere you can run Python. Prefect handles the heavy lifting for you out of the box: automatic state tracking, failure handling, real-time monitoring, and more.
### 
[​](https://docs.prefect.io/v3/get-started#essential-features)
Essential features
Feature | Description  
---|---  
**Pythonic** | Write workflows in native Python—no DSLs, YAML, or special syntax. Full support for type hints, async/await, and modern Python patterns. Use your existing IDE, debugger, and testing tools.  
**State & Recovery** | Robust state management that tracks success, failure, and retry states. Resume interrupted runs from the last successful point, and cache expensive computations to avoid unnecessary rework.  
**Flexible & Portable Execution** | Start flows locally for easy development, then deploy them anywhere—from a single process to containers, Kubernetes, or cloud services—without locking into a vendor. Infrastructure is defined by code (not just configuration), making it simple to scale or change environments.  
**Event-Driven** | Trigger flows on schedules, external events, or via API. Pause flows for human intervention or approval. Chain flows together based on states, conditions, or any custom logic.  
**Dynamic Runtime** | Create tasks dynamically at runtime based on actual data or conditions. Easily spawn new tasks and branches during execution for truly data-driven workflows.  
**Modern UI** | Real-time flow run monitoring, logging, and state tracking through an intuitive interface. View dependency graphs and DAGs automatically—just run your flow and open the UI.  
**CI/CD First** | Test and simulate flows like normal Python code, giving you fast feedback during development. Integrate seamlessly into your existing CI/CD pipeline for automated testing and deployment.  
## 
[​](https://docs.prefect.io/v3/get-started#quickstart)
Quickstart
## [Quickstart Quickly create your first deployable workflow tracked by Prefect. ](https://docs.prefect.io/v3/get-started/quickstart)## [Install Prefect Install Prefect and get connected to Prefect Cloud or a self-hosted server. ](https://docs.prefect.io/v3/get-started/install)## [Upgrade to Prefect 3 Upgrade from Prefect 2 to Prefect 3 to get the latest features and performance enhancements. ](https://docs.prefect.io/v3/how-to-guides/migrate/upgrade-to-prefect-3)
## 
[​](https://docs.prefect.io/v3/get-started#how-to-guides)
How-to guides
## [Build workflows Learn how to write and customize your Prefect workflows with tasks and flows. ](https://docs.prefect.io/v3/how-to-guides/workflows/write-and-run)## [Deploy workflows Deploy and manage your workflows as Prefect deployments. ](https://docs.prefect.io/v3/how-to-guides/deployments/create-deployments)## [Configure infrastructure Deploy your workflows to specific infrastructure platforms. ](https://docs.prefect.io/v3/how-to-guides/deployment_infra/managed)## [Set up automations Work with events, triggers, and automations to build reactive workflows. ](https://docs.prefect.io/v3/how-to-guides/automations/creating-automations)## [Configure Prefect Configure your Prefect environment, secrets, and variables. ](https://docs.prefect.io/v3/how-to-guides/configuration/manage-settings)## [Use Prefect Cloud Set up and manage your Prefect Cloud account. ](https://docs.prefect.io/v3/how-to-guides/cloud/connect-to-cloud)
## 
[​](https://docs.prefect.io/v3/get-started#advanced)
Advanced
## [Interactive workflows Build interactive workflows that can pause and receive input. ](https://docs.prefect.io/v3/advanced/interactive)## [Platform engineering Use Prefect as a platform for your teams’ data pipelines. ](https://docs.prefect.io/v3/advanced/infrastructure-as-code)## [Extend Prefect Extend Prefect with custom blocks and API integrations. ](https://docs.prefect.io/v3/advanced/api-client)
## 
[​](https://docs.prefect.io/v3/get-started#examples)
Examples
Check out the gallery of [examples](https://docs.prefect.io/v3/examples/index) to see Prefect in action.
## 
[​](https://docs.prefect.io/v3/get-started#mini-history-of-prefect)
Mini-history of Prefect
**2018-2021:** Our story begins in 2018, when we introduced the idea that workflow orchestration should be Pythonic. Inspired by distributed tools like Dask, and building on the experience of our founder, Jeremiah Lowin (a PMC member of Apache Airflow), we created a system based on simple Python decorators for tasks and flows. But what made Prefect truly special was our introduction of task mapping—a feature that would later become foundational to our dynamic execution capabilities (and eventually imitated by other orchestration SDKs). **2022:** Prefect’s 2.0 release became inevitable once we recognized that real-world workflows don’t always fit into neat, pre-planned DAG structures: sometimes you need to update a job definition based on runtime information, for example by skipping a branch of your workflow. So we removed a key constraint that workflows be written explicitly as DAGs, fully embracing native Python control flow—if/else conditionals, while loops-everything that makes Python…Python. **2023-present:** With our release of Prefect 3.0 in 2024, we fully embraced these dynamic patterns by open-sourcing our events and automations backend, allowing users to natively represent event-driven workflows and gain additional observability into their execution. Prefect 3.0 also unlocked a leap forward in performance, improving the runtime overhead of Prefect by up to 90%.
## 
[​](https://docs.prefect.io/v3/get-started#join-our-community)
Join our community
Join Prefect’s vibrant [community of nearly 30,000 engineers](https://docs.prefect.io/contribute/index) to learn with others and share your knowledge!
Was this page helpful?
YesNo
[Install Prefect](https://docs.prefect.io/v3/get-started/install)
Configuration
# How to manage settings
### 
[​](https://docs.prefect.io/v3/how-to-guides/configuration/manage-settings#view-current-configuration)
View current configuration
To view all available settings and their active values from the command line, run:
Copy
```
prefect config view --show-defaults

```

These settings are type-validated and you may verify your setup at any time with:
Copy
```
prefect config validate

```

### 
[​](https://docs.prefect.io/v3/how-to-guides/configuration/manage-settings#configure-settings-for-the-active-profile)
Configure settings for the active profile
To update a setting for the active profile, run:
Copy
```
prefect config set <setting>=<value>

```

For example, to set the `PREFECT_API_URL` setting to `http://127.0.0.1:4200/api`, run:
Copy
```
prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api

```

To restore the default value for a setting, run:
Copy
```
prefect config unset <setting>

```

For example, to restore the default value for the `PREFECT_API_URL` setting, run:
Copy
```
prefect config unset PREFECT_API_URL

```

### 
[​](https://docs.prefect.io/v3/how-to-guides/configuration/manage-settings#create-a-new-profile)
Create a new profile
To create a new profile, run:
Copy
```
prefect profile create <profile>

```

To switch to a new profile, run:
Copy
```
prefect profile use <profile>

```

### 
[​](https://docs.prefect.io/v3/how-to-guides/configuration/manage-settings#configure-settings-for-a-project)
Configure settings for a project
To configure settings for a project, create a `prefect.toml` or `.env` file in the project directory and add the settings with the values you want to use. For example, to configure the `PREFECT_API_URL` setting to `http://127.0.0.1:4200/api`, create a `.env` file with the following content:
Copy
```
PREFECT_API_URL=http://127.0.0.1:4200/api

```

To configure the `PREFECT_API_URL` setting to `http://127.0.0.1:4200/api` in a `prefect.toml` file, create a `prefect.toml` file with the following content:
Copy
```
api.url = "http://127.0.0.1:4200/api"

```

Refer to the [setting concept guide](https://docs.prefect.io/v3/concepts/settings-and-profiles) for more information on how to configure settings and the [settings reference guide](https://docs.prefect.io/v3/api-ref/settings-ref) for more information on the available settings.
### 
[​](https://docs.prefect.io/v3/how-to-guides/configuration/manage-settings#configure-temporary-settings-for-a-process)
Configure temporary settings for a process
To configure temporary settings for a process, set an environment variable with the name matching the setting you want to configure. For example, to configure the `PREFECT_API_URL` setting to `http://127.0.0.1:4200/api` for a process, set the `PREFECT_API_URL` environment variable to `http://127.0.0.1:4200/api`.
Copy
```
export PREFECT_API_URL=http://127.0.0.1:4200/api

```

You can use this to run a command with the temporary setting:
Copy
```
PREFECT_LOGGING_LEVEL=DEBUG python my_script.py

```

Was this page helpful?
YesNo
[Share configuration between workflows](https://docs.prefect.io/v3/how-to-guides/configuration/variables)[Create Automations](https://docs.prefect.io/v3/how-to-guides/automations/creating-automations)
Workflow Infrastructure
# How to run flows on Prefect Managed infrastructure
Learn how Prefect runs deployments on Prefect’s infrastructure.
Flows that run with this work pool do not require a worker or cloud provider account—Prefect handles the infrastructure and code execution for you. Managed execution is a great option for users who want to get started quickly, with no infrastructure setup.
## 
[​](https://docs.prefect.io/v3/how-to-guides/deployment_infra/managed#create-a-managed-deployment)
Create a managed deployment
  1. Create a new work pool of type Prefect Managed in the UI or the CLI. Use this command to create a new work pool using the CLI:
Copy
```
prefect work-pool create my-managed-pool --type prefect:managed

```

  2. Create a deployment using the flow `deploy` method or `prefect.yaml`. Specify the name of your managed work pool, as shown in this example that uses the `deploy` method:
managed-execution.py
Copy
```
from prefect import flow
if __name__ == "__main__":
    flow.from_source(
        source="https://github.com/prefecthq/demo.git",
        entrypoint="flow.py:my_flow",
    ).deploy(
        name="test-managed-flow",
        work_pool_name="my-managed-pool",
    )

```

  3. With your [CLI authenticated to your Prefect Cloud workspace](https://docs.prefect.io/v3/how-to-guides/cloud/manage-users/api-keys), run the script to create your deployment:
Copy
```
python managed-execution.py

```

  4. Run the deployment from the UI or from the CLI. This process runs a flow on remote infrastructure without any infrastructure setup, starting a worker, or requiring a cloud provider account.


## 
[​](https://docs.prefect.io/v3/how-to-guides/deployment_infra/managed#add-dependencies)
Add dependencies
Prefect can install Python packages in the container that runs your flow at runtime. Specify these dependencies in the **Pip Packages** field in the UI, or by configuring `job_variables={"pip_packages": ["pandas", "prefect-aws"]}` in your deployment creation like this:
Copy
```
from prefect import flow
if __name__ == "__main__":
    flow.from_source(
        source="https://github.com/prefecthq/demo.git",
        entrypoint="flow.py:my_flow",
    ).deploy(
        name="test-managed-flow",
        work_pool_name="my-managed-pool",
        job_variables={"pip_packages": ["pandas", "prefect-aws"]}
    )

```

Alternatively, you can create a `requirements.txt` file and reference it in your [prefect.yaml pull step](https://docs.prefect.io/v3/deploy/infrastructure-concepts/prefect-yaml#utility-steps).
## 
[​](https://docs.prefect.io/v3/how-to-guides/deployment_infra/managed#networking)
Networking
### 
[​](https://docs.prefect.io/v3/how-to-guides/deployment_infra/managed#static-outbound-ip-addresses)
Static Outbound IP addresses
Flows running on Prefect Managed infrastructure can be assigned static IP addresses on outbound traffic, which will pose as the `source` address for requests interacting with your system or database. Include these `source` addresses in your firewall rules to explicitly allow flows to communicate with your environment.
  * `184.73.85.134`
  * `52.4.218.198`
  * `44.217.117.74`


### 
[​](https://docs.prefect.io/v3/how-to-guides/deployment_infra/managed#images)
Images
Managed execution requires that you run an official Prefect Docker image, such as `prefecthq/prefect:3-latest`. However, as noted above, you can install Python package dependencies at runtime. If you need to use your own image, we recommend using another type of work pool.
### 
[​](https://docs.prefect.io/v3/how-to-guides/deployment_infra/managed#code-storage)
Code storage
You must store flow code in an accessible remote location. Prefect supports git-based cloud providers such as GitHub, Bitbucket, or GitLab. Remote block-based storage is also supported, so S3, GCS, and Azure Blob are additional code storage options.
### 
[​](https://docs.prefect.io/v3/how-to-guides/deployment_infra/managed#resources-%26-limitations)
Resources & Limitations
All flow runs receive:
  * 4vCPUs
  * 16GB of RAM
  * 128GB of ephemeral storage

Maximum flow run time is limited to 24 hours. Every account has a compute usage limit per workspace that resets monthly. Compute used is defined as the duration between compute startup and teardown, rounded up to the nearest minute. This is approximately, although not exactly, the duration of a flow run going from `PENDING` to `COMPLETED`.
## 
[​](https://docs.prefect.io/v3/how-to-guides/deployment_infra/managed#next-steps)
Next steps
Read more about creating deployments in [Run flows in Docker containers](https://docs.prefect.io/v3/how-to-guides/deployment_infra/docker). For more control over your infrastructure, such as the ability to run custom Docker images, [serverless push work pools](https://docs.prefect.io/v3/how-to-guides/deployment_infra/serverless) are a good option.
Was this page helpful?
YesNo
[Run Flows in Local Processes](https://docs.prefect.io/v3/how-to-guides/deployment_infra/run-flows-in-local-processes)[Run flows on serverless compute](https://docs.prefect.io/v3/how-to-guides/deployment_infra/serverless)
Deployments
# How to create deployments
Learn how to create deployments via the CLI, Python, or Terraform.
There are several ways to create deployments, each catering to different organizational needs.
## 
[​](https://docs.prefect.io/v3/how-to-guides/deployments/create-deployments#create-deployments-with-static-infrastructure)
Create deployments with static infrastructure
### 
[​](https://docs.prefect.io/v3/how-to-guides/deployments/create-deployments#create-a-deployment-with-serve)
Create a deployment with `serve`
This is the simplest way to get started with deployments:
Copy
```
from prefect import flow
@flow
def my_flow():
    print("Hello, Prefect!")
if __name__ == "__main__":
    my_flow.serve(name="my-first-deployment", cron="* * * * *")

```

The `serve` method creates a deployment from your flow and immediately begins listening for scheduled runs to execute. Providing `cron="* * * * *"` to `.serve` associates a schedule with your flow so it will run every minute of every day.
## 
[​](https://docs.prefect.io/v3/how-to-guides/deployments/create-deployments#create-deployments-with-dynamic-infrastructure)
Create deployments with dynamic infrastructure
For more configuration, you can create a deployment that uses a work pool. Reasons to create a work-pool based deployment include:
  * Wanting to run your flow on dynamically provisioned infrastructure
  * Needing more control over the execution environment on a per-flow run basis
  * Creating an infrastructure template to use across deployments

Work pools are popular with data platform teams because they allow you to manage infrastructure configuration across an organization. Prefect offers two options for creating deployments with [dynamic infrastructure](https://docs.prefect.io/v3/deploy/infrastructure-concepts/work-pools):
  * [Deployments created with the Python SDK](https://docs.prefect.io/v3/deploy/infrastructure-concepts/deploy-via-python)
  * [Deployments created with a YAML file](https://docs.prefect.io/v3/deploy/infrastructure-concepts/prefect-yaml)


### 
[​](https://docs.prefect.io/v3/how-to-guides/deployments/create-deployments#create-a-deployment-with-deploy)
Create a deployment with `deploy`
To define a deployment with a Python script, use the `flow.deploy` method. Here’s an example of a deployment that uses a work pool and bakes the code into a Docker image.
Copy
```
from prefect import flow
@flow
def my_flow():
    print("Hello, Prefect!")
if __name__ == "__main__":
    my_flow.deploy(
        name="my-second-deployment",
        work_pool_name="my-work-pool",
        image="my-image",
        push=False,
        cron="* * * * *",
    )

```

To learn more about the `deploy` method, see [Deploy flows with Python](https://docs.prefect.io/v3/deploy/infrastructure-concepts/deploy-via-python).
### 
[​](https://docs.prefect.io/v3/how-to-guides/deployments/create-deployments#create-a-deployment-with-a-yaml-file)
Create a deployment with a YAML file
If you’d rather take a declarative approach to defining a deployment through a YAML file, use a [`prefect.yaml` file](https://docs.prefect.io/v3/deploy/infrastructure-concepts/prefect-yaml). Prefect provides an interactive CLI that walks you through creating a `prefect.yaml` file:
Copy
```
prefect deploy

```

The result is a `prefect.yaml` file for deployment creation. The file contains `build`, `push`, and `pull` steps for building a Docker image, pushing code to a Docker registry, and pulling code at runtime. Learn more about creating deployments with a YAML file in [Define deployments with YAML](https://docs.prefect.io/v3/deploy/infrastructure-concepts/prefect-yaml). Prefect also provides [CI/CD options](https://docs.prefect.io/v3/deploy/infrastructure-concepts/deploy-ci-cd) for automatically creating YAML-based deployments.
### 
[​](https://docs.prefect.io/v3/how-to-guides/deployments/create-deployments#create-a-deployment-with-terraform)
Create a deployment with Terraform
You can manage deployments with the [Terraform provider for Prefect](https://registry.terraform.io/providers/PrefectHQ/prefect/latest/docs/resources/deployment).
### 
[​](https://docs.prefect.io/v3/how-to-guides/deployments/create-deployments#create-a-deployment-with-the-api)
Create a deployment with the API
You can manage deployments with the [Prefect API](https://app.prefect.cloud/api/docs#tag/Deployments).
**Choosing between deployment methods** For many cases, `serve` is sufficient for scheduling and orchestration. The work pool / worker paradigm via `.deploy()` or `prefect deploy` can be great for complex infrastructure requirements and isolated flow run environments. You are not locked into one method and can combine approaches as needed.
Was this page helpful?
YesNo
[Limit concurrent task runs with tags](https://docs.prefect.io/v3/how-to-guides/workflows/tag-based-concurrency-limits)[Trigger ad-hoc deployment runs](https://docs.prefect.io/v3/how-to-guides/deployments/run-deployments)
Migrate
# How to upgrade to Prefect 3.0
Learn how to upgrade from Prefect 2.x to Prefect 3.0.
Prefect 3.0 introduces a number of enhancements to the OSS product: a new events & automations backend for event-driven workflows and observability, improved runtime performance, autonomous task execution and a streamlined caching layer based on transactional semantics. The majority of these enhancements maintain compatibility with most Prefect 2.0 workflows, but there are a few caveats that you may need to adjust for. To learn more about the enhanced performance and new features, see [What’s new in Prefect 3.0](https://docs.prefect.io/v3/get-started/whats-new-prefect-3). For the majority of users, upgrading to Prefect 3.0 will be a seamless process that requires few or no code changes. This guide highlights key changes that you may need to consider when upgrading.
**Prefect 2.0** refers to the 2.x lineage of the open source prefect package, and **Prefect 3.0** refers exclusively to the 3.x lineage of the prefect package. Neither version is strictly tied to any aspect of Prefect’s commercial product, [Prefect Cloud](https://docs.prefect.io/v3/how-to-guides/cloud/connect-to-cloud).
## 
[​](https://docs.prefect.io/v3/how-to-guides/migrate/upgrade-to-prefect-3#quickstart)
Quickstart
To upgrade to Prefect 3.0, run:
Copy
```
pip install -U prefect

```

If you self-host a Prefect server, run this command to update your database:
Copy
```
prefect server database upgrade

```

If you use a Prefect integration or extra, remember to upgrade it as well. For example:
Copy
```
pip install -U 'prefect[aws]'

```

## 
[​](https://docs.prefect.io/v3/how-to-guides/migrate/upgrade-to-prefect-3#upgrade-notes)
Upgrade notes
### 
[​](https://docs.prefect.io/v3/how-to-guides/migrate/upgrade-to-prefect-3#pydantic-v2)
Pydantic V2
Prefect 3.0 is built with Pydantic 2.0 for improved performance. All Prefect objects will automatically upgrade, but if you use custom Pydantic models for flow parameters or custom blocks, you’ll need to ensure they are compatible with Pydantic 2.0. You can continue to use Pydantic 1.0 models in your own code if they do not interact directly with Prefect. Refer to [Pydantic’s migration guide](https://docs.pydantic.dev/latest/migration/) for detailed information on necessary changes.
We recommend pausing all deployment schedules prior to upgrading. Because of differences in Pydantic datetime handling that affect the scheduler’s idempotency logic, there is a small risk of the scheduler duplicating runs in its first loop.
### 
[​](https://docs.prefect.io/v3/how-to-guides/migrate/upgrade-to-prefect-3#module-location-and-name-changes)
Module location and name changes
Some less-commonly used modules have been renamed, reorganized, or removed for clarity. The old import paths will continue to be supported for 6 months, but emit deprecation warnings. You can look at the [deprecation code](https://github.com/PrefectHQ/prefect/blob/main/src/prefect/_internal/compatibility/migration.py) to see a full list of affected paths.
### 
[​](https://docs.prefect.io/v3/how-to-guides/migrate/upgrade-to-prefect-3#async-tasks-in-synchronous-flows)
Async tasks in synchronous flows
In Prefect 2.0, it was possible to call native `async` tasks from synchronous flows, a pattern that is not normally supported in Python. Prefect 3.0.0 removes this behavior to reduce complexity and potential issues and edge cases. If you relied on asynchronous tasks in synchronous flows, you must either make your flow asynchronous or use a task runner that supports asynchronous execution.
### 
[​](https://docs.prefect.io/v3/how-to-guides/migrate/upgrade-to-prefect-3#flow-final-states)
Flow final states
In Prefect 2.0, the final state of a flow run was influenced by the states of its task runs; if any task run failed, the flow run was marked as failed. In Prefect 3.0, the final state of a flow run is entirely determined by:
  1. The `return` value of the flow function (same as in Prefect 2.0):
     * Literal values are considered successful.
     * Any explicit `State` that is returned will be considered the final state of the flow run. If an iterable of `State` objects is returned, all must be `Completed` for the flow run to be considered `Completed`. If any are `Failed`, the flow run will be marked as `Failed`.
  2. Whether the flow function allows an exception to `raise`:
     * Exceptions that are allowed to propagate will result in a `Failed` state.
     * Exceptions suppressed with `raise_on_failure=False` will not affect the flow run state.

This change means that task failures within a flow do not automatically cause the flow run to fail unless they affect the flow’s return value or raise an uncaught exception.
When migrating from Prefect 2.0 to Prefect 3, be aware that flows may now complete successfully even if they contain failed tasks, unless you explicitly handle task failures.
To ensure your flow fails when critical tasks fail, consider these approaches:
  1. Allow task exceptions to propagate by not using `raise_on_failure=False`.
  2. Use `return_state=True` and explicitly check task states to conditionally `raise` the underlying exception or return a failed state.
  3. Use try/except blocks to handle task failures and return appropriate states.


#### 
[​](https://docs.prefect.io/v3/how-to-guides/migrate/upgrade-to-prefect-3#examples)
Examples
Allow Unhandled Exceptions
Use return_state
Use try/except
Copy
```
from prefect import flow, task
@task
def failing_task():
    raise ValueError("Task failed")
@flow
def my_flow():
    failing_task()  # Exception propagates, causing flow failure
try:
    my_flow()
except ValueError as e:
    print(f"Flow failed: {e}")  # Output: Flow failed: Task failed

```

Choose the strategy that best fits your specific use case and error handling requirements.
* * *
### 
[​](https://docs.prefect.io/v3/how-to-guides/migrate/upgrade-to-prefect-3#futures-interface)
Futures interface
PrefectFutures now have a standard synchronous interface, with an asynchronous one [planned soon](https://github.com/PrefectHQ/prefect/issues/15008).
### 
[​](https://docs.prefect.io/v3/how-to-guides/migrate/upgrade-to-prefect-3#automatic-task-caching)
Automatic task caching
Prefect 3.0 introduces a powerful idempotency engine. By default, tasks in a flow run are automatically cached if they are called more than once with the same inputs. If you rely on tasks with side effects, this may result in surprising behavior. To disable caching, pass `cache_policy=None` to your task.
### 
[​](https://docs.prefect.io/v3/how-to-guides/migrate/upgrade-to-prefect-3#workers)
Workers
In Prefect 2.0, agents were deprecated in favor of next-generation workers. Workers are now standard in Prefect 3. For detailed information on upgrading from agents to workers, please refer to our [upgrade guide](https://docs-3.prefect.io/v3/resources/upgrade-agents-to-workers).
### 
[​](https://docs.prefect.io/v3/how-to-guides/migrate/upgrade-to-prefect-3#resolving-common-gotchas)
Resolving common gotchas
#### 
[​](https://docs.prefect.io/v3/how-to-guides/migrate/upgrade-to-prefect-3#attributeerror%3A-coroutine-object-has-no-attribute-%3Csome-attribute%3E)
`AttributeError: 'coroutine' object has no attribute <some attribute>`
When within an asynchronous task or flow context, if you do **not** `await` an asynchronous function or method, this error will be raised when you try to use the object. To fix it, `await` the asynchronous function or method or use `_sync=True`. For example, `Block`’s `load` method is asynchronous in an async context:
Incorrect
Correct
Also Correct
Copy
```
from prefect.blocks.system import Secret
async def my_async_function():
    my_secret = Secret.load("my-secret")
    print(my_secret.get()) # AttributeError: 'coroutine' object has no attribute 'get'

```

Similarly, if you never use an un-awaited coroutine, you may see a warning like this:
Copy
```
RuntimeWarning: coroutine 'some_async_callable' was never awaited
...
RuntimeWarning: Enable tracemalloc to get the object allocation traceback

```

This is the same problem as above, and you may `await` the coroutine to fix it or use `_sync=True`.
#### 
[​](https://docs.prefect.io/v3/how-to-guides/migrate/upgrade-to-prefect-3#typeerror%3A-object-%3Csome-type%3E-cant-be-used-in-await-expression)
`TypeError: object <some Type> can't be used in 'await' expression`
This error occurs when using the `await` keyword before an object that is not a coroutine. To fix it, remove the `await`. For example, `my_task.submit(...)` is _always_ synchronous in Prefect 3.x:
Incorrect
Correct
Copy
```
from prefect import flow, task
@task
async def my_task():
    pass
@flow
async def my_flow():
    future = await my_task.submit() # TypeError: object PrefectConcurrentFuture can't be used in 'await' expression

```

See the [Futures interface section](https://docs.prefect.io/v3/how-to-guides/migrate/upgrade-to-prefect-3#futures-interface) for more information on this particular gotcha.
#### 
[​](https://docs.prefect.io/v3/how-to-guides/migrate/upgrade-to-prefect-3#typeerror%3A-flow-deploy-got-an-unexpected-keyword-argument-schedule)
`TypeError: Flow.deploy() got an unexpected keyword argument 'schedule'`
In Prefect 3.0, the `schedule` argument has been removed in favor of the `schedules` argument. This applies to both the `Flow.serve` and `Flow.deploy` methods.
Prefect 2.0
Prefect 3.0
Copy
```
from datetime import timedelta
from prefect import flow
from prefect.client.schemas.schedules import IntervalSchedule
@flow
def my_flow():
    pass
my_flow.serve(
    name="my-flow",
    schedule=IntervalSchedule(interval=timedelta(minutes=1))
)

```

Was this page helpful?
YesNo
[Migrate from Airflow](https://docs.prefect.io/v3/how-to-guides/migrate/airflow)[Upgrade from agents to workers](https://docs.prefect.io/v3/how-to-guides/migrate/upgrade-agents-to-workers)
Platform
# How to build deployments via CI/CD
CI/CD resources for working with Prefect.
Many organizations deploy Prefect workflows through their CI/CD process. Each organization has their own unique CI/CD setup, but a common pattern is to use CI/CD to manage Prefect [deployments](https://docs.prefect.io/v3/concepts/deployments). Combining Prefect’s deployment features with CI/CD tools enables efficient management of flow code updates, scheduling changes, and container builds. This guide uses [GitHub Actions](https://docs.github.com/en/actions) to implement a CI/CD process, but these concepts are generally applicable across many CI/CD tools. Note that Prefect’s primary ways for creating deployments, a `.deploy` flow method or a `prefect.yaml` configuration file, are both designed for building and pushing images to a Docker registry.
## 
[​](https://docs.prefect.io/v3/advanced/deploy-ci-cd#get-started-with-github-actions-and-prefect)
Get started with GitHub Actions and Prefect
In this example, you’ll write a GitHub Actions workflow that runs each time you push to your repository’s `main` branch. This workflow builds and pushes a Docker image containing your flow code to Docker Hub, then deploys the flow to Prefect Cloud.
### 
[​](https://docs.prefect.io/v3/advanced/deploy-ci-cd#repository-secrets)
Repository secrets
Your CI/CD process must be able to authenticate with Prefect to deploy flows. Deploy flows securely and non-interactively in your CI/CD process by saving your `PREFECT_API_URL` and `PREFECT_API_KEY` [as secrets in your repository’s settings](https://docs.github.com/en/actions/security-guides/using-secrets-in-github-actions). This allows them to be accessed in your CI/CD runner’s environment without exposing them in any scripts or configuration files. In this scenario, deploying flows involves building and pushing Docker images, so add `DOCKER_USERNAME` and `DOCKER_PASSWORD` as secrets to your repository as well. Create secrets for GitHub Actions in your repository under **Settings - > Secrets and variables -> Actions -> New repository secret**:
### 
[​](https://docs.prefect.io/v3/advanced/deploy-ci-cd#write-a-github-workflow)
Write a GitHub workflow
To deploy your flow through GitHub Actions, you need a workflow YAML file. GitHub looks for workflow YAML files in the `.github/workflows/` directory in the root of your repository. In their simplest form, GitHub workflow files are made up of triggers and jobs. The `on:` trigger is set to run the workflow each time a push occurs on the `main` branch of the repository. The `deploy` job is comprised of four `steps`:
  * **`Checkout`**clones your repository into the GitHub Actions runner so you can reference files or run scripts from your repository in later steps.
  * **`Log in to Docker Hub`**authenticates to DockerHub so your image can be pushed to the Docker registry in your DockerHub account.[docker/login-action](https://github.com/docker/login-action) is an existing GitHub action maintained by Docker. `with:` passes values into the Action, similar to passing parameters to a function.
  * **`Setup Python`**installs your selected version of Python.
  * **`Prefect Deploy`**installs the dependencies used in your flow, then deploys your flow.`env:` makes the `PREFECT_API_KEY` and `PREFECT_API_URL` secrets from your repository available as environment variables during this step’s execution.

For reference, the examples below live in their respective branches of [this repository](https://github.com/prefecthq/cicd-example).
  * .deploy
  * prefect.yaml


Copy
```
.
| -- .github/
|   |-- workflows/
|       |-- deploy-prefect-flow.yaml
|-- flow.py
|-- requirements.txt

```

`flow.py`
Copy
```
from prefect import flow
@flow(log_prints=True)
def hello():
  print("Hello!")
if __name__ == "__main__":
    hello.deploy(
        name="my-deployment",
        work_pool_name="my-work-pool",
        image="my_registry/my_image:my_image_tag",
    )

```

`.github/workflows/deploy-prefect-flow.yaml`
Copy
```
name: Deploy Prefect flow
on:
  push:
    branches:
      - main
jobs:
  deploy:
    name: Deploy
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
      - name: Log in to Docker Hub
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Prefect Deploy
        env:
          PREFECT_API_KEY: ${{ secrets.PREFECT_API_KEY }}
          PREFECT_API_URL: ${{ secrets.PREFECT_API_URL }}
        run: |
          pip install -r requirements.txt
          python flow.py

```

### 
[​](https://docs.prefect.io/v3/advanced/deploy-ci-cd#run-a-github-workflow)
Run a GitHub workflow
After pushing commits to your repository, GitHub automatically triggers a run of your workflow. Monitor the status of running and completed workflows from the **Actions** tab of your repository. View the logs from each workflow step as they run. The `Prefect Deploy` step includes output about your image build and push, and the creation/update of your deployment.
Copy
```
Successfully built image '***/cicd-example:latest'
Successfully pushed image '***/cicd-example:latest'
Successfully created/updated all deployments!
                Deployments
|-----------------------------------------|
| Name                | Status    Details |
|---------------------|---------|---------|
| hello/my-deployment | applied |         |
|-----------------------------------------|

```

## 
[​](https://docs.prefect.io/v3/advanced/deploy-ci-cd#advanced-example)
Advanced example
In more complex scenarios, CI/CD processes often need to accommodate several additional considerations to enable a smooth development workflow:
  * Making code available in different environments as it advances through stages of development
  * Handling independent deployment of distinct groupings of work, as in a monorepo
  * Efficiently using build time to avoid repeated work

This [example repository](https://github.com/prefecthq/cicd-example-workspaces) addresses each of these considerations with a combination of Prefect’s and GitHub’s capabilities.
### 
[​](https://docs.prefect.io/v3/advanced/deploy-ci-cd#deploy-to-multiple-workspaces)
Deploy to multiple workspaces
The deployment processes to run are automatically selected when changes are pushed, depending on two conditions:
Copy
```
on:
  push:
    branches:
      - stg
      - main
    paths:
      - "project_1/**"

```

  * **`branches:`**- which branch has changed. This ultimately selects which Prefect workspace a deployment is created or updated in. In this example, changes on the`stg` branch deploy flows to a staging workspace, and changes on the `main` branch deploy flows to a production workspace.
  * **`paths:`**- which project folders’ files have changed. Since each project folder contains its own flows, dependencies, and`prefect.yaml` , it represents a complete set of logic and configuration that can deploy independently. Each project in this repository gets its own GitHub Actions workflow YAML file.

The `prefect.yaml` file in each project folder depends on environment variables dictated by the selected job in each CI/CD workflow; enabling external code storage for Prefect deployments that is clearly separated across projects and environments.
Copy
```
  .
  |--- cicd-example-workspaces-prod  # production bucket
  |   |--- project_1
  |   |---project_2
  |---cicd-example-workspaces-stg  # staging bucket
      |--- project_1
      |---project_2

```

Deployments in this example use S3 for code storage. So it’s important that push steps place flow files in separate locations depending upon their respective environment and project—so no deployment overwrites another deployment’s files.
### 
[​](https://docs.prefect.io/v3/advanced/deploy-ci-cd#caching-build-dependencies)
Caching build dependencies
Since building Docker images and installing Python dependencies are essential parts of the deployment process, it’s useful to rely on caching to skip repeated build steps. The `setup-python` action offers [caching options](https://github.com/actions/setup-python#caching-packages-dependencies) so Python packages do not have to be downloaded on repeat workflow runs.
Copy
```
- name: Setup Python
  uses: actions/setup-python@v5
  with:
    python-version: "3.12"
    cache: "pip"

```

The `build-push-action` for building Docker images also offers [caching options for GitHub Actions](https://docs.docker.com/build/cache/backends/gha/). If you are not using GitHub, other remote [cache backends](https://docs.docker.com/build/cache/backends/) are available as well.
Copy
```
- name: Build and push
  id: build-docker-image
  env:
      GITHUB_SHA: ${{ steps.get-commit-hash.outputs.COMMIT_HASH }}
  uses: docker/build-push-action@v5
  with:
    context: ${{ env.PROJECT_NAME }}/
    push: true
    tags: ${{ secrets.DOCKER_USERNAME }}/${{ env.PROJECT_NAME }}:${{ env.GITHUB_SHA }}-stg
    cache-from: type=gha
    cache-to: type=gha,mode=max

```

Copy
```
importing cache manifest from gha:***
DONE 0.1s
[internal] load build context
transferring context: 70B done
DONE 0.0s
[2/3] COPY requirements.txt requirements.txt
CACHED
[3/3] RUN pip install -r requirements.txt
CACHED

```

## 
[​](https://docs.prefect.io/v3/advanced/deploy-ci-cd#prefect-github-actions)
Prefect GitHub Actions
Prefect provides its own GitHub Actions for [authentication](https://github.com/PrefectHQ/actions-prefect-auth) and [deployment creation](https://github.com/PrefectHQ/actions-prefect-deploy). These actions simplify deploying with CI/CD when using `prefect.yaml`, especially in cases where a repository contains flows used in multiple deployments across multiple Prefect Cloud workspaces. Here’s an example of integrating these actions into the workflow above:
Copy
```
name: Deploy Prefect flow
on:
  push:
    branches:
      - main
jobs:
  deploy:
    name: Deploy
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
      - name: Log in to Docker Hub
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Prefect Auth
        uses: PrefectHQ/actions-prefect-auth@v1
        with:
          prefect-api-key: ${{ secrets.PREFECT_API_KEY }}
          prefect-workspace: ${{ secrets.PREFECT_WORKSPACE }}
      - name: Run Prefect Deploy
        uses: PrefectHQ/actions-prefect-deploy@v4
        with:
          deployment-names: my-deployment
          requirements-file-paths: requirements.txt

```

## 
[​](https://docs.prefect.io/v3/advanced/deploy-ci-cd#authenticate-to-other-docker-image-registries)
Authenticate to other Docker image registries
The `docker/login-action` GitHub Action supports pushing images to a wide variety of image registries. For example, if you are storing Docker images in AWS Elastic Container Registry, you can add your ECR registry URL to the `registry` key in the `with:` part of the action and use an `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` as your `username` and `password`.
Copy
```
- name: Login to ECR
  uses: docker/login-action@v3
  with:
    registry: <aws-account-number>.dkr.ecr.<region>.amazonaws.com
    username: ${{ secrets.AWS_ACCESS_KEY_ID }}
    password: ${{ secrets.AWS_SECRET_ACCESS_KEY }}

```

## 
[​](https://docs.prefect.io/v3/advanced/deploy-ci-cd#further-reading)
Further reading
You can manage resources with the [Terraform provider for Prefect](https://registry.terraform.io/providers/PrefectHQ/prefect/latest/docs/guides/getting-started).
Was this page helpful?
YesNo
[Maintain your Prefect database](https://docs.prefect.io/v3/advanced/database-maintenance)[Manage resources with Terraform and Helm](https://docs.prefect.io/v3/advanced/infrastructure-as-code)
Manage accounts
# How to manage API keys
Create an API key to access Prefect Cloud from a local execution environment.
API keys enable you to authenticate a local environment to work with Prefect Cloud. If you run `prefect cloud login` from your CLI, you can authenticate through your browser or by pasting an API key. Authenticating through the browser directs you to an authorization page. After you grant approval to connect, you’re redirected to the CLI and the API key is saved to your local [Prefect profile](https://docs.prefect.io/v3/develop/settings-and-profiles). If you choose to authenticate by pasting an API key, you must create an API key in the Prefect Cloud UI first.
## 
[​](https://docs.prefect.io/v3/how-to-guides/cloud/manage-users/api-keys#create-an-api-key)
Create an API key
  1. Select the account icon at the bottom-left corner of the UI.
  2. Select **API Keys**. The page displays a list of previously generated keys, and allows you to create or delete API keys.
  3. Select the **+** button to create a new API key. Provide a name for the key and an expiration date.


Copy the key to a secure location since an API key cannot be revealed again in the UI after it is generated.
## 
[​](https://docs.prefect.io/v3/how-to-guides/cloud/manage-users/api-keys#log-in-to-prefect-cloud-with-an-api-key)
Log in to Prefect Cloud with an API Key
Copy
```
prefect cloud login -k '<my-api-key>'

```

Alternatively, if you don’t have a CLI available - for example, if you are connecting to Prefect Cloud within a remote serverless environment - set the `PREFECT_API_KEY` environment variable. For more information see [Connect to Prefect Cloud](https://docs.prefect.io/v3/manage/cloud/connect-to-cloud).
## 
[​](https://docs.prefect.io/v3/how-to-guides/cloud/manage-users/api-keys#service-account-api-keys-pro-enterprise)
Service account API keys (Pro) (Enterprise)
Service accounts are a feature of Prefect Cloud [Pro and Enterprise tier plans](https://www.prefect.io/pricing) that enable you to create a Prefect Cloud API key that is not associated with a user account. Service accounts are useful for configuring API access for running workers, or executing flow runs on remote infrastructure. Events and logs for flow runs in those environments are associated with the service account rather than a user. Manage or revoke API access by configuring or removing the service account without disrupting user access. See [service accounts](https://docs.prefect.io/v3/how-to-guides/cloud/manage-users/service-accounts) for more information.
Was this page helpful?
YesNo
[Manage account roles](https://docs.prefect.io/v3/how-to-guides/cloud/manage-users/manage-roles)[Configure single sign-on](https://docs.prefect.io/v3/how-to-guides/cloud/manage-users/configure-sso)
Deployments
# How to deploy flows with Python
Learn how to use the Python SDK to deploy flows to run in work pools.
As an alternative to defining deployments in a `prefect.yaml` file, you can use the Python SDK to create deployments with [dynamic infrastructure](https://docs.prefect.io/v3/concepts/work-pools). This can be more flexible if you have to programmatically gather configuration at deployment time.
## 
[​](https://docs.prefect.io/v3/how-to-guides/deployments/deploy-via-python#when-to-use-flow-deploy-instead-of-flow-serve)
When to use `flow.deploy` instead of `flow.serve`
The `flow.serve` method is one [simple way to create a deployment with the Python SDK](https://docs.prefect.io/v3/how-to-guides/deployment_infra/run-flows-in-local-processes#serve-a-flow). It’s ideal when you have readily available, static infrastructure for your flow runs and you don’t need the dynamic dispatch of infrastructure per flow run offered by work pools. However, you might want to consider using `flow.deploy` to associate your flow with a work pool that enables dynamic dispatch of infrastructure per flow run for the following reasons:
  1. **Cost optimization:** Dynamic infrastructure can help reduce costs by scaling resources up or down based on demand.
  2. **Resource scarcity:** If you have limited persistent infrastructure, dynamic provisioning can help manage resource allocation more efficiently.
  3. **Varying workloads:** For workflows with inconsistent resource needs, dynamic infrastructure can adapt to changing requirements.
  4. **Cloud-native deployments:** When working with cloud providers that offer serverless or auto-scaling options.

Let’s explore how to create a deployment using the Python SDK and leverage dynamic infrastructure through work pools.
## 
[​](https://docs.prefect.io/v3/how-to-guides/deployments/deploy-via-python#prerequisites)
Prerequisites
Before deploying your flow using `flow.deploy`, ensure you have the following:
  1. **A running Prefect server or Prefect Cloud workspace:** You can either run a Prefect server locally or use a Prefect Cloud workspace. To start a local server, run `prefect server start`. To use Prefect Cloud, sign up for an account at [app.prefect.cloud](https://app.prefect.cloud) and follow the [Connect to Prefect Cloud](https://docs.prefect.io/v3/manage/cloud/connect-to-cloud) guide.
  2. **A Prefect flow:** You should have a flow defined in your Python script. If you haven’t created a flow yet, refer to the [Write Workflows](https://docs.prefect.io/v3/how-to-guides/workflows/write-and-run) guide.
  3. **A work pool:** You need a work pool to manage the infrastructure for running your flow. If you haven’t created a work pool, you can do so through the Prefect UI or using the Prefect CLI. For more information, see the [Work Pools](https://docs.prefect.io/v3/concepts/work-pools) guide. For examples in this guide, we’ll use a Docker work pool created by running:
Copy
```
prefect work-pool create --type docker my-work-pool

```

  4. **Docker:** Docker will be used to build and store the image containing your flow code. You can download and install Docker from the [official Docker website](https://www.docker.com/get-started/). If you don’t want to use Docker, you can see other options in the [Use remote code storage](https://docs.prefect.io/v3/how-to-guides/deployments/deploy-via-python#use-remote-code-storage) section.
  5. **(Optional) A Docker registry:** While not strictly necessary for local development, having an account with a Docker registry (such as Docker Hub) is recommended for storing and sharing your Docker images.

With these prerequisites in place, you’re ready to deploy your flow using `flow.deploy`.
## 
[​](https://docs.prefect.io/v3/how-to-guides/deployments/deploy-via-python#deploy-a-flow-with-flow-deploy)
Deploy a flow with `flow.deploy`
To deploy a flow using `flow.deploy` and Docker, follow these steps:
1
Write a flow
Ensure your flow is defined in a Python file. Let’s use a simple example:
example.py
Copy
```
from prefect import flow
@flow(log_prints=True)
def my_flow(name: str = "world"):
    print(f"Hello, {name}!")

```

2
Add deployment configuration
Add a call to `flow.deploy` to tell Prefect how to deploy your flow.
example.py
Copy
```
from prefect import flow
@flow(log_prints=True)
def my_flow(name: str = "world"):
    print(f"Hello, {name}!")
if __name__ == "__main__":
    my_flow.deploy(
        name="my-deployment",
        work_pool_name="my-work-pool",
        image="my-registry.com/my-docker-image:my-tag",
        push=False # switch to True to push to your image registry
    )

```

3
Deploy!
Run your script to deploy your flow.
Copy
```
python example.py

```

Running this script will:
  1. Build a Docker image containing your flow code and dependencies.
  2. Create a deployment associated with the specified work pool and image.


Building a Docker image for our flow allows us to have a consistent environment for our flow to run in. Workers for our work pool will use the image to run our flow. In this example, we set `push=False` to skip pushing the image to a registry. This is useful for local development, and you can push your image to a registry such as Docker Hub in a production environment.
**Where’s the Dockerfile?** In the above example, we didn’t specify a Dockerfile. By default, Prefect will generate a Dockerfile for us that copies the flow code into an image and installs any additional dependencies.If you want to write and use your own Dockerfile, you can do so by passing a `dockerfile` parameter to `flow.deploy`.
### 
[​](https://docs.prefect.io/v3/how-to-guides/deployments/deploy-via-python#trigger-a-run)
Trigger a run
Now that we have our flow deployed, we can trigger a run via either the Prefect CLI or UI. First, we need to start a worker to run our flow:
Copy
```
prefect worker start --pool my-work-pool

```

Then, we can trigger a run of our flow using the Prefect CLI:
Copy
```
prefect deployment run 'my-flow/my-deployment'

```

After a few seconds, you should see logs from your worker showing that the flow run has started and see the state update in the UI.
## 
[​](https://docs.prefect.io/v3/how-to-guides/deployments/deploy-via-python#deploy-with-a-schedule)
Deploy with a schedule
To deploy a flow with a schedule, you can use one of the following options:
  * `interval` Defines the interval at which the flow should run. Accepts an integer or float value representing the number of seconds between runs or a `datetime.timedelta` object.
Show Example
interval.py
Copy
```
from datetime import timedelta
from prefect import flow
@flow(log_prints=True)
def my_flow(name: str = "world"):
    print(f"Hello, {name}!")
if __name__ == "__main__":
    my_flow.deploy(
        name="my-deployment",
        work_pool_name="my-work-pool",
        image="my-registry.com/my-docker-image:my-tag",
        push=False,
        # Run once a minute
        interval=timedelta(minutes=1)
    )

```

  * `cron` Defines when a flow should run using a cron string.
Show Example
cron.py
Copy
```
from prefect import flow
@flow(log_prints=True)
def my_flow(name: str = "world"):
    print(f"Hello, {name}!")
if __name__ == "__main__":
    my_flow.deploy(
        name="my-deployment",
        work_pool_name="my-work-pool",
        image="my-registry.com/my-docker-image:my-tag",
        push=False,
        # Run once a day at midnight
        cron="0 0 * * *"
    )

```

  * `rrule` Defines a complex schedule using an `rrule` string.
Show Example
rrule.py
Copy
```
from prefect import flow
@flow(log_prints=True)
def my_flow(name: str = "world"):
    print(f"Hello, {name}!")
if __name__ == "__main__":
    my_flow.deploy(
        name="my-deployment",
        work_pool_name="my-work-pool",
        image="my-registry.com/my-docker-image:my-tag",
        push=False,
        # Run every weekday at 9 AM
        rrule="FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;BYHOUR=9;BYMINUTE=0"
    )

```

  * `schedules` Defines multiple schedules for a deployment. This option provides flexibility for:
    * Setting up various recurring schedules
    * Implementing complex scheduling logic
    * Applying timezone offsets to schedules
Show Example
schedules.py
Copy
```
from datetime import datetime, timedelta
from prefect import flow
from prefect.schedules import Interval
@flow(log_prints=True)
def my_flow(name: str = "world"):
    print(f"Hello, {name}!")
if __name__ == "__main__":
    my_flow.deploy(
        name="my-deployment",
        work_pool_name="my-work-pool",
        image="my-registry.com/my-docker-image:my-tag",
        push=False,
        # Run every 10 minutes starting from January 1, 2023
        # at 00:00 Central Time
        schedules=[
            Interval(
                timedelta(minutes=10),
                anchor_date=datetime(2023, 1, 1, 0, 0),
                timezone="America/Chicago"
            )
        ]
    )

```

Learn more about schedules [here](https://docs.prefect.io/v3/how-to-guides/deployments/create-schedules).


## 
[​](https://docs.prefect.io/v3/how-to-guides/deployments/deploy-via-python#use-remote-code-storage)
Use remote code storage
In addition to storing your code in a Docker image, Prefect also supports deploying your code to remote storage. This approach allows you to store your flow code in a remote location, such as a Git repository or cloud storage service. Using remote storage for your code has several advantages:
  1. Faster iterations: You can update your flow code without rebuilding Docker images.
  2. Reduced storage requirements: You don’t need to store large Docker images for each code version.
  3. Flexibility: You can use different storage backends based on your needs and infrastructure.

Using an existing remote Git repository like GitHub, GitLab, or Bitbucket works really well as remote code storage because:
  1. Your code is already there.
  2. You can deploy to multiple environments via branches and tags.
  3. You can roll back to previous versions of your flows.

To deploy using remote storage, you’ll need to specify where your code is stored by using `flow.from_source` to first load your flow code from a remote location. Here’s an example of loading a flow from a Git repository and deploying it:
git-deploy.py
Copy
```
from prefect import flow
if __name__ == "__main__":
    flow.from_source(
        source="https://github.com/username/repository.git",
        entrypoint="path/to/your/flow.py:your_flow_function"
    ).deploy(
        name="my-deployment",
        work_pool_name="my-work-pool",
    )

```

The `source` parameter can accept a variety of remote storage options including:
  * Git repositories
  * S3 buckets (using the `s3://` scheme)
  * Google Cloud Storage buckets (using the `gs://` scheme)
  * Azure Blob Storage (using the `az://` scheme)

The `entrypoint` parameter is the path to the flow function within your repository combined with the name of the flow function. Learn more about remote code storage [here](https://docs.prefect.io/v3/deploy/infrastructure-concepts/store-flow-code).
## 
[​](https://docs.prefect.io/v3/how-to-guides/deployments/deploy-via-python#set-default-parameters)
Set default parameters
You can set default parameters for a deployment using the `parameters` keyword argument in `flow.deploy`.
default-parameters.py
Copy
```
from prefect import flow
@flow
def my_flow(name: str = "world"):
    print(f"Hello, {name}!")
if __name__ == "__main__":
    my_flow.deploy(
        name="my-deployment",
        work_pool_name="my-work-pool",
        # Will print "Hello, Marvin!" by default instead of "Hello, world!"
        parameters={"name": "Marvin"},
        image="my-registry.com/my-docker-image:my-tag",
        push=False,
    )

```

Note these parameters can still be overridden on a per-deployment basis.
## 
[​](https://docs.prefect.io/v3/how-to-guides/deployments/deploy-via-python#set-job-variables)
Set job variables
You can set default job variables for a deployment using the `job_variables` keyword argument in `flow.deploy`. The job variables provided will override the values set on the work pool.
job-variables.py
Copy
```
import os
from prefect import flow
@flow
def my_flow():
    name = os.getenv("NAME", "world")
    print(f"Hello, {name}!")
if __name__ == "__main__":
    my_flow.deploy(
        name="my-deployment",
        work_pool_name="my-work-pool",
        # Will print "Hello, Marvin!" by default instead of "Hello, world!"
        job_variables={"env": {"NAME": "Marvin"}},
        image="my-registry.com/my-docker-image:my-tag",
        push=False,
    )

```

Job variables can be used to customize environment variables, resources limits, and other infrastructure options, allowing fine-grained control over your infrastructure on a per-deployment or per-flow-run basis. Any variable defined in the base job template of the associated work pool can be overridden by a job variable. You can learn more about job variables [here](https://docs.prefect.io/v3/how-to-guides/deployments/customize-job-variables).
## 
[​](https://docs.prefect.io/v3/how-to-guides/deployments/deploy-via-python#deploy-multiple-flows)
Deploy multiple flows
To deploy multiple flows at once, use the `deploy` function.
multi-deploy.py
Copy
```
from prefect import flow, deploy
@flow
def my_flow_1():
    print("I'm number one!")
@flow
def my_flow_2():
    print("Always second...")
if __name__ == "__main__":
    deploy(
        # Use the `to_deployment` method to specify configuration
        #specific to each deployment
        my_flow_1.to_deployment("my-deployment-1"),
        my_flow_2.to_deployment("my-deployment-2"),
        # Specify shared configuration for both deployments
        image="my-registry.com/my-docker-image:my-tag",
        push=False,
        work_pool_name="my-work-pool",
    )

```

When we run the above script, it will build a single Docker image for both deployments. This approach offers the following benefits:
  * Saves time and resources by avoiding redundant image builds.
  * Simplifies management by maintaining a single image for multiple flows.


## 
[​](https://docs.prefect.io/v3/how-to-guides/deployments/deploy-via-python#further-reading)
Further reading
  * [Work Pools](https://docs.prefect.io/v3/concepts/work-pools)
  * [Store Flow Code](https://docs.prefect.io/v3/deploy/infrastructure-concepts/store-flow-code)
  * [Customize Infrastructure](https://docs.prefect.io/v3/how-to-guides/deployments/customize-job-variables)
  * [Schedules](https://docs.prefect.io/v3/how-to-guides/deployments/create-schedules)
  * [Write Workflows](https://docs.prefect.io/v3/how-to-guides/workflows/write-and-run)


Was this page helpful?
YesNo
[Create Deployment Schedules](https://docs.prefect.io/v3/how-to-guides/deployments/create-schedules)[Define deployments with YAML](https://docs.prefect.io/v3/how-to-guides/deployments/prefect-yaml)
Deployments
# How to define deployments with YAML
Use YAML to schedule and trigger flow runs and manage your code and deployments.
The `prefect.yaml` file is a YAML file describing base settings for your deployments, procedural steps for preparing deployments, and instructions for preparing the execution environment for a deployment run. Initialize your deployment configuration, which creates the `prefect.yaml` file, with the CLI command `prefect init` in any directory or repository that stores your flow code.
**Deployment configuration recipes** Prefect ships with many off-the-shelf “recipes” that allow you to get started with more structure within your `prefect.yaml` file. Run `prefect init` to be prompted with available recipes in your installation. You can provide a recipe name in your initialization command with the `--recipe` flag, otherwise Prefect will attempt to guess an appropriate recipe based on the structure of your working directory (for example if you initialize within a `git` repository, Prefect will use the `git` recipe).
The `prefect.yaml` file contains:
  * deployment configuration for deployments created from this file
  * default instructions for how to build and push any necessary code artifacts (such as Docker images)
  * default instructions for pulling a deployment in remote execution environments (for example, cloning a GitHub repository).

You can override any deployment configuration through options available on the `prefect deploy` CLI command when creating a deployment.
**`prefect.yaml`file flexibility** In older versions of Prefect, this file must be in the root of your repository or project directory and named `prefect.yaml`. With Prefect 3, this file can be located in a directory outside the project or a subdirectory inside the project. It can be named differently if the filename ends in `.yaml`. You can have multiple `prefect.yaml` files with the same name in different directories.By default, `prefect deploy` uses a `prefect.yaml` file in the project’s root directory. To use a custom deployment configuration file, supply the new `--prefect-file` CLI argument when running the `deploy` command from the root of your project directory:`prefect deploy --prefect-file path/to/my_file.yaml`
The base structure for `prefect.yaml` looks like this:
Copy
```
# generic metadata
prefect-version: null
name: null
# preparation steps
build: null
push: null
# runtime steps
pull: null
# deployment configurations
deployments:
- # base metadata
  name: null
  version: null
  tags: []
  description: null
  schedule: null
  # flow-specific fields
  entrypoint: null
  parameters: {}
  # infra-specific fields
  work_pool:
    name: null
    work_queue_name: null
    job_variables: {}

```

The metadata fields are always pre-populated for you. These fields are for bookkeeping purposes only. The other sections are pre-populated based on recipe; if no recipe is provided, Prefect attempts to guess an appropriate one based on local configuration. You can create deployments with the CLI command `prefect deploy` without altering the `deployments` section of your `prefect.yaml` file. The `prefect deploy` command helps in deployment creation through interactive prompts. The `prefect.yaml` file facilitates version-controlling your deployment configuration and managing multiple deployments.
## 
[​](https://docs.prefect.io/v3/how-to-guides/deployments/prefect-yaml#deployment-actions)
Deployment actions
Deployment actions defined in your `prefect.yaml` file control the lifecycle of the creation and execution of your deployments. The three actions available are `build`, `push`, and `pull`. `pull` is the only required deployment action. It defines how Prefect pulls your deployment in remote execution environments. Each action is defined as a list of steps executed in sequence. Each step has the following format:
Copy
```
section:
- prefect_package.path.to.importable.step:
  id: "step-id" # optional
  requires: "pip-installable-package-spec" # optional
  kwarg1: value
  kwarg2: more-values

```

Every step optionally provides a `requires` field. Prefect uses this to auto-install if the step is not found in the current environment. Each step can specify an `id` to reference outputs in future steps. The additional fields map directly to Python keyword arguments to the step function. Within a given section, steps always run in their order within the `prefect.yaml` file.
**Deployment instruction overrides** You can override `build`, `push`, and `pull` sections on a per-deployment basis; define `build`, `push`, and `pull` fields within a deployment definition in the `prefect.yaml` file.The `prefect deploy` command uses any `build`, `push`, or `pull` instructions from the deployment’s definition in the `prefect.yaml` file.This capability is useful for multiple deployments that require different deployment instructions.
### 
[​](https://docs.prefect.io/v3/how-to-guides/deployments/prefect-yaml#the-build-action)
The build action
Use the build section of `prefect.yaml` to specify setup steps or dependencies, (like creating a Docker image), required to run your deployments. If you initialize with the Docker recipe, you are prompted to provide required information, such as image name and tag:
Copy
```
prefect init --recipe docker
>> image_name: < insert image name here >
>> tag: < insert image tag here >

```

**Use`--field` to avoid the interactive experience**We recommend that you only initialize a recipe when first creating your deployment structure. Then store your configuration files within version control. Sometimes you may need to initialize programmatically and avoid the interactive prompts. To do this, provide all required fields for your recipe using the `--field` flag:
Copy
```
prefect init --recipe docker \
    --field image_name=my-repo/my-image \
    --field tag=my-tag

```

Copy
```
build:
- prefect_docker.deployments.steps.build_docker_image:
  requires: prefect-docker>=0.3.0
  image_name: my-repo/my-image
  tag: my-tag
  dockerfile: auto

```

Once you confirm that these fields are set to their desired values, this step automatically builds a Docker image with the provided name and tag and pushes it to the repository referenced by the image name. This step produces optional fields for future steps, or within `prefect.yaml` as template values. We recommend using a templated `{{ image }}` within `prefect.yaml` (specifically in the work pool’s `job_variables` section). By avoiding hardcoded values, the build step and deployment specification won’t have mismatched values.
**Some steps require Prefect integrations** In the build step example above, you relied on the `prefect-docker` package; in cases that deal with external services, additional required packages are auto-installed for you.
**Pass output to downstream steps** Each deployment action can be composed of multiple steps. For example, to build a Docker image tagged with the current commit hash, use the `run_shell_script` step and feed the output into the `build_docker_image` step:
Copy
```
build:
- prefect.deployments.steps.run_shell_script:
    id: get-commit-hash
    script: git rev-parse --short HEAD
    stream_output: false
- prefect_docker.deployments.steps.build_docker_image:
    requires: prefect-docker
    image_name: my-image
    image_tag: "{{ get-commit-hash.stdout }}"
    dockerfile: auto

```

The `id` field is used in the `run_shell_script` step to reference its output in the next step.
### 
[​](https://docs.prefect.io/v3/how-to-guides/deployments/prefect-yaml#the-push-action)
The push action
The push section is most critical for situations where code is not stored on persistent filesystems or in version control. In this scenario, code is often pushed and pulled from a Cloud storage bucket (for example, S3, GCS, Azure Blobs). The push section allows users to specify and customize the logic for pushing this code repository to arbitrary remote locations. For example, a user who stores their code in an S3 bucket and relies on default worker settings for its runtime environment could use the `s3` recipe:
Copy
```
prefect init --recipe s3
>> bucket: < insert bucket name here >

```

In your newly created`prefect.yaml` file, you should find that the `push` and `pull` sections have been templated out as follows:
Copy
```
push:
- prefect_aws.deployments.steps.push_to_s3:
    id: push-code
    requires: prefect-aws>=0.3.0
    bucket: my-bucket
    folder: project-name
    credentials: null
pull:
- prefect_aws.deployments.steps.pull_from_s3:
    requires: prefect-aws>=0.3.0
    bucket: my-bucket
    folder: "{{ push-code.folder }}"
    credentials: null

```

The bucket is populated with the provided value (which also could have been provided with the `--field` flag); note that the `folder` property of the `push` step is a template—the `pull_from_s3` step outputs both a `bucket` value as well as a `folder` value for the template downstream steps. This helps you keep your steps consistent across edits. As discussed above, if you use [blocks](https://docs.prefect.io/v3/concepts/blocks), you can template the credentials section with a block reference for secure and dynamic credentials access:
Copy
```
push:
- prefect_aws.deployments.steps.push_to_s3:
    requires: prefect-aws>=0.3.0
    bucket: my-bucket
    folder: project-name
    credentials: "{{ prefect.blocks.aws-credentials.dev-credentials }}"

```

Anytime you run `prefect deploy`, this `push` section executes upon successful completion of your `build` section.
### 
[​](https://docs.prefect.io/v3/how-to-guides/deployments/prefect-yaml#the-pull-action)
The pull action
The pull section is the most important section within the `prefect.yaml` file. It contains instructions for preparing your flows for a deployment run. These instructions execute each time a deployment in this folder is run through a worker. There are three main types of steps that typically show up in a `pull` section:
  * `set_working_directory`: this step sets the working directory for the process prior to importing your flow
  * `git_clone`: this step clones the provided repository on the provided branch
  * `pull_from_{cloud}`: this step pulls the working directory from a Cloud storage location (for example, S3)


**Use block and variable references** All [block and variable references](https://docs.prefect.io/v3/how-to-guides/deployments/prefect-yaml#templating-options) within your pull step will remain unresolved until runtime and are pulled each time your deployment runs. This avoids storing sensitive information insecurely; it also allows you to manage certain types of configuration from the API and UI without having to rebuild your deployment every time.
Below is an example of how to use an existing `GitHubCredentials` block to clone a private GitHub repository:
Copy
```
pull:
- prefect.deployments.steps.git_clone:
    repository: https://github.com/org/repo.git
    credentials: "{{ prefect.blocks.github-credentials.my-credentials }}"

```

Alternatively, you can specify a `BitBucketCredentials` or `GitLabCredentials` block to clone from Bitbucket or GitLab. In lieu of a credentials block, you can also provide a GitHub, GitLab, or Bitbucket token directly to the ‘access_token` field. Use a Secret block to do this securely:
Copy
```
pull:
- prefect.deployments.steps.git_clone:
    repository: https://bitbucket.org/org/repo.git
    access_token: "{{ prefect.blocks.secret.bitbucket-token }}"

```

## 
[​](https://docs.prefect.io/v3/how-to-guides/deployments/prefect-yaml#utility-steps)
Utility steps
Use utility steps within a build, push, or pull action to assist in managing the deployment lifecycle:
  * `run_shell_script` allows for the execution of one or more shell commands in a subprocess, and returns the standard output and standard error of the script. This step is useful for scripts that require execution in a specific environment, or those which have specific input and output requirements. Note that setting `stream_output: true` for `run_shell_script` writes the output and error to stdout in the execution environment, which will not be sent to the Prefect API.

Here is an example of retrieving the short Git commit hash of the current repository to use as a Docker image tag:
Copy
```
build:
- prefect.deployments.steps.run_shell_script:
    id: get-commit-hash
    script: git rev-parse --short HEAD
    stream_output: false
- prefect_docker.deployments.steps.build_docker_image:
    requires: prefect-docker>=0.3.0
    image_name: my-image
    tag: "{{ get-commit-hash.stdout }}"
    dockerfile: auto

```

**Provided environment variables are not expanded by default** To expand environment variables in your shell script, set `expand_env_vars: true` in your `run_shell_script` step. For example:
Copy
```
- prefect.deployments.steps.run_shell_script:
    id: get-user
    script: echo $USER
    stream_output: true
    expand_env_vars: true

```

Without `expand_env_vars: true`, the above step returns a literal string `$USER` instead of the current user.
  * `pip_install_requirements` installs dependencies from a `requirements.txt` file within a specified directory.

Here is an example of installing dependencies from a `requirements.txt` file after cloning:
Copy
```
pull:
- prefect.deployments.steps.git_clone:
    id: clone-step # needed to be referenced in subsequent steps
    repository: https://github.com/org/repo.git
- prefect.deployments.steps.pip_install_requirements:
    directory: "{{ clone-step.directory }}" # `clone-step` is a user-provided `id` field
    requirements_file: requirements.txt

```

Here is an example that retrieves an access token from a third party key vault and uses it in a private clone step:
Copy
```
pull:
- prefect.deployments.steps.run_shell_script:
    id: get-access-token
    script: az keyvault secret show --name <secret name> --vault-name <secret vault> --query "value" --output tsv
    stream_output: false
- prefect.deployments.steps.git_clone:
    repository: https://bitbucket.org/samples/deployments.git
    branch: master
    access_token: "{{ get-access-token.stdout }}"

```

You can also run custom steps by packaging them. In the example below, `retrieve_secrets` is a custom python module packaged into the default working directory of a Docker image (which is /opt/prefect by default). `main` is the function entry point, which returns an access token (for example, `return {"access_token": access_token}`) like the preceding example, but utilizing the Azure Python SDK for retrieval.
Copy
```
- retrieve_secrets.main:
    id: get-access-token
- prefect.deployments.steps.git_clone:
    repository: https://bitbucket.org/samples/deployments.git
    branch: master
    access_token: '{{ get-access-token.access_token }}'

```

## 
[​](https://docs.prefect.io/v3/how-to-guides/deployments/prefect-yaml#templating-options)
Templating options
Values that you place within your `prefect.yaml` file can reference dynamic values in several different ways:
  * **step outputs** : every step of both `build` and `push` produce named fields such as `image_name`; you can reference these fields within `prefect.yaml` and `prefect deploy` will populate them with each call. References must be enclosed in double brackets and in `"{{ field_name }}"` format
  * **blocks** : you can reference [Prefect blocks](https://docs.prefect.io/v3/concepts/blocks) with the `{{ prefect.blocks.block_type.block_slug }}` syntax. It is highly recommended that you use block references for any sensitive information (such as a GitHub access token or any credentials) to avoid hardcoding these values in plaintext
  * **variables** : you can reference [Prefect variables](https://docs.prefect.io/v3/concepts/variables) with the `{{ prefect.variables.variable_name }}` syntax. Use variables to reference non-sensitive, reusable pieces of information such as a default image name or a default work pool name.
  * **environment variables** : you can also reference environment variables with the special syntax `{{ $MY_ENV_VAR }}`. This is especially useful for referencing environment variables that are set at runtime.

Here’s a `prefect.yaml` file as an example:
Copy
```
build:
- prefect_docker.deployments.steps.build_docker_image:
    id: build-image
    requires: prefect-docker>=0.6.0
    image_name: my-repo/my-image
    tag: my-tag
    dockerfile: auto
push:
- prefect_docker.deployments.steps.push_docker_image:
    requires: prefect-docker>=0.6.0
    image_name: my-repo/my-image
    tag: my-tag
    credentials: "{{ prefect.blocks.docker-registry-credentials.dev-registry }}"
deployments:
- # base metadata
  name: null
  version: "{{ build-image.tag }}"
  tags:
  - "{{ $my_deployment_tag }}"
  - "{{ prefect.variables.some_common_tag }}"
  description: null
  schedule: null
  concurrency_limit: null
  # flow-specific fields
  entrypoint: null
  parameters: {}
  # infra-specific fields
  work_pool:
    name: "my-k8s-work-pool"
    work_queue_name: null
    job_variables:
      image: "{{ build-image.image }}"
      cluster_config: "{{ prefect.blocks.kubernetes-cluster-config.my-favorite-config }}"

```

So long as your `build` steps produce fields called `image_name` and `tag`, every time you deploy a new version of our deployment, the `{{ build-image.image }}` variable is dynamically populated with the relevant values.
**Docker step** The most commonly used build step is `prefect_docker.deployments.steps.build_docker_image` which produces both the `image_name` and `tag` fields.
A `prefect.yaml` file can have multiple deployment configurations that control the behavior of several deployments. You can manage these deployments independently of one another, allowing you to deploy the same flow with different configurations in the same codebase.
## 
[​](https://docs.prefect.io/v3/how-to-guides/deployments/prefect-yaml#work-with-multiple-deployments-with-prefect-yaml)
Work with multiple deployments with prefect.yaml
Prefect supports multiple deployment declarations within the `prefect.yaml` file. This method of declaring multiple deployments supports version control for all deployments through a single command. Add new deployment declarations to the `prefect.yaml` file with a new entry to the `deployments` list. Each deployment declaration must have a unique `name` field to select deployment declarations when using the `prefect deploy` command.
When using a `prefect.yaml` file that is in another directory or differently named, the value for the deployment `entrypoint` must be relative to the root directory of the project.
For example, consider the following `prefect.yaml` file:
Copy
```
build: ...
push: ...
pull: ...
deployments:
  - name: deployment-1
    entrypoint: flows/hello.py:my_flow
    parameters:
        number: 42,
        message: Don't panic!
    work_pool:
        name: my-process-work-pool
        work_queue_name: primary-queue
  - name: deployment-2
    entrypoint: flows/goodbye.py:my_other_flow
    work_pool:
        name: my-process-work-pool
        work_queue_name: secondary-queue
  - name: deployment-3
    entrypoint: flows/hello.py:yet_another_flow
    work_pool:
        name: my-docker-work-pool
        work_queue_name: tertiary-queue

```

This file has three deployment declarations, each referencing a different flow. Each deployment declaration has a unique `name` field and can be deployed individually with the `--name` flag when deploying. For example, to deploy `deployment-1`, run:
Copy
```
prefect deploy --name deployment-1

```

To deploy multiple deployments, provide multiple `--name` flags:
Copy
```
prefect deploy --name deployment-1 --name deployment-2

```

To deploy multiple deployments with the same name, prefix the deployment name with its flow name:
Copy
```
prefect deploy --name my_flow/deployment-1 --name my_other_flow/deployment-1

```

To deploy all deployments, use the `--all` flag:
Copy
```
prefect deploy --all

```

To deploy deployments that match a pattern, run:
Copy
```
prefect deploy -n my-flow/* -n *dev/my-deployment -n dep*prod

```

The above command deploys:
  * all deployments from the flow `my-flow`
  * all flows ending in `dev` with a deployment named `my-deployment`
  * all deployments starting with `dep` and ending in `prod`.


### 
[​](https://docs.prefect.io/v3/how-to-guides/deployments/prefect-yaml#non-interactive-deployment)
Non-interactive deployment
For CI/CD pipelines and automated environments, use the `--no-prompt` flag to skip interactive prompts:
Copy
```
prefect --no-prompt deploy --name my-deployment

```

This prevents the command from hanging on prompts and will fail clearly if required information is missing.
**CLI Options When deploying multiple deployments** When deploying more than one deployment with a single `prefect deploy` command, any additional attributes provided are ignored.To provide overrides to a deployment through the CLI, you must deploy that deployment individually.
### 
[​](https://docs.prefect.io/v3/how-to-guides/deployments/prefect-yaml#reuse-configuration-across-deployments)
Reuse configuration across deployments
Because a `prefect.yaml` file is a standard YAML file, you can use [YAML aliases](https://yaml.org/spec/1.2.2/#71-alias-nodes) to reuse configuration across deployments. This capability allows multiple deployments to share the work pool configuration, deployment actions, or other configurations. Declare a YAML alias with the `&{alias_name}` syntax and insert that alias elsewhere in the file with the `*{alias_name}` syntax. When aliasing YAML maps, you can override specific fields of the aliased map with the `<<: *{alias_name}` syntax and adding additional fields below. We recommend adding a `definitions` section to your `prefect.yaml` file at the same level as the `deployments` section to store your aliases. For example:
Copy
```
build: ...
push: ...
pull: ...
definitions:
    work_pools:
        my_docker_work_pool: &my_docker_work_pool
            name: my-docker-work-pool
            work_queue_name: default
            job_variables:
                image: "{{ build-image.image }}"
    schedules:
        every_ten_minutes: &every_10_minutes
            interval: 600
    actions:
        docker_build: &docker_build
            - prefect_docker.deployments.steps.build_docker_image: &docker_build_config
                id: build-image
                requires: prefect-docker>=0.3.0
                image_name: my-example-image
                tag: dev
                dockerfile: auto
        docker_push: &docker_push
            - prefect_docker.deployments.steps.push_docker_image: &docker_push_config
                requires: prefect-docker>=0.6.0
                image_name: my-example-image
                tag: dev
                credentials: "{{ prefect.blocks.docker-registry-credentials.dev-registry }}"
deployments:
  - name: deployment-1
    entrypoint: flows/hello.py:my_flow
    schedule: *every_10_minutes
    parameters:
        number: 42,
        message: Don't panic!
    work_pool: *my_docker_work_pool
    build: *docker_build # Uses the full docker_build action with no overrides
    push: *docker_push
  - name: deployment-2
    entrypoint: flows/goodbye.py:my_other_flow
    work_pool: *my_docker_work_pool
    build:
        - prefect_docker.deployments.steps.build_docker_image:
            <<: *docker_build_config # Uses the docker_build_config alias and overrides the dockerfile field
            dockerfile: Dockerfile.custom
    push: *docker_push
  - name: deployment-3
    entrypoint: flows/hello.py:yet_another_flow
    schedule: *every_10_minutes
    work_pool:
        name: my-process-work-pool
        work_queue_name: primary-queue

```

In the above example, YAML aliases reuse work pool, schedule, and build configuration across multiple deployments:
  * `deployment-1` and `deployment-2` use the same work pool configuration
  * `deployment-1` and `deployment-3` use the same schedule
  * `deployment-1` and `deployment-2` use the same build deployment action, but `deployment-2` overrides the `dockerfile` field to use a custom Dockerfile


## 
[​](https://docs.prefect.io/v3/how-to-guides/deployments/prefect-yaml#deployment-declaration-reference)
Deployment declaration reference
### 
[​](https://docs.prefect.io/v3/how-to-guides/deployments/prefect-yaml#deployment-fields)
Deployment fields
These are fields you can add to each deployment declaration. Property | Description  
---|---  
`name` | The name to give to the created deployment. Used with the `prefect deploy` command to create or update specific deployments.  
`version` | An optional version for the deployment.  
`tags` | A list of strings to assign to the deployment as tags.  
`description` | An optional description for the deployment.  
`schedule` | An optional [schedule](https://docs.prefect.io/v3/how-to-guides/deployments/create-schedules) to assign to the deployment. Fields for this section are documented in the [Schedule Fields](https://docs.prefect.io/v3/how-to-guides/deployments/prefect-yaml#schedule-fields) section.  
`concurrency_limit` | An optional [deployment concurrency limit](https://docs.prefect.io/v3/deploy/index#concurrency-limiting). Fields for this section are documented in the [Concurrency Limit Fields](https://docs.prefect.io/v3/how-to-guides/deployments/prefect-yaml#concurrency-limit-fields) section.  
`triggers` | An optional array of [triggers](https://docs.prefect.io/v3/how-to-guides/automations/creating-deployment-triggers) to assign to the deployment  
`entrypoint` | Required path to the `.py` file containing the flow you want to deploy (relative to the root directory of your development folder) combined with the name of the flow function. In the format `path/to/file.py:flow_function_name`.  
`parameters` | Optional default values to provide for the parameters of the deployed flow. Should be an object with key/value pairs.  
`enforce_parameter_schema` | Boolean flag that determines whether the API should validate the parameters passed to a flow run against the parameter schema generated for the deployed flow.  
`work_pool` | Information of where to schedule flow runs for the deployment. Fields for this section are documented in the [Work Pool Fields](https://docs.prefect.io/v3/how-to-guides/deployments/prefect-yaml#work-pool-fields) section.  
### 
[​](https://docs.prefect.io/v3/how-to-guides/deployments/prefect-yaml#schedule-fields)
Schedule fields
These are fields you can add to a deployment declaration’s `schedule` section. Property | Description  
---|---  
`interval` | Number of seconds indicating the time between flow runs. Cannot use them in conjunction with `cron` or `rrule`.  
`anchor_date` | Datetime string indicating the starting or “anchor” date to begin the schedule. If no `anchor_date` is supplied, the current UTC time is used. Can only use with `interval`.  
`timezone` | String name of a time zone, used to enforce localization behaviors like DST boundaries. See the [IANA Time Zone Database](https://www.iana.org/time-zones) for valid time zones.  
`cron` | A valid cron string. Cannot use in conjunction with `interval` or `rrule`.  
`day_or` | Boolean indicating how croniter handles day and day_of_week entries. Must use with `cron`. Defaults to `True`.  
`rrule` | String representation of an RRule schedule. See the [`rrulestr` examples](https://dateutil.readthedocs.io/en/stable/rrule.html#rrulestr-examples) for syntax. Cannot used them in conjunction with `interval` or `cron`.  
### 
[​](https://docs.prefect.io/v3/how-to-guides/deployments/prefect-yaml#concurrency-limit-fields)
Concurrency limit fields
These are fields you can add to a deployment declaration’s `concurrency_limit` section. Property | Description  
---|---  
`limit` | The maximum number of concurrent flow runs for the deployment.  
`collision_strategy` | Configure the behavior for runs once the concurrency limit is reached. Options are `ENQUEUE`, and `CANCEL_NEW`. Defaults to `ENQUEUE`.  
### 
[​](https://docs.prefect.io/v3/how-to-guides/deployments/prefect-yaml#work-pool-fields)
Work pool fields
These are fields you can add to a deployment declaration’s `work_pool` section. Property | Description  
---|---  
`name` | The name of the work pool to schedule flow runs in for the deployment.  
`work_queue_name` | The name of the work queue within the specified work pool to schedule flow runs in for the deployment. If not provided, the default queue for the specified work pool is used.  
`job_variables` | Values used to override the default values in the specified work pool’s [base job template](https://docs.prefect.io/v3/concepts/work-pools#base-job-template). Maps directly to a created deployments `infra_overrides` attribute.  
### 
[​](https://docs.prefect.io/v3/how-to-guides/deployments/prefect-yaml#deployment-mechanics)
Deployment mechanics
Anytime you run `prefect deploy` in a directory that contains a `prefect.yaml` file, the following actions take place in order:
  * The `prefect.yaml` file load. First, the `build` section loads and all variable and block references resolve. The steps then run in the order provided.
  * Next, the `push` section loads and all variable and block references resolve; the steps within this section then run in the order provided.
  * Next, the `pull` section is templated with any step outputs but _is not run_. Block references are _not_ hydrated for security purposes: they are always resolved at runtime.
  * Next, all variable and block references resolve with the deployment declaration. All flags provided through the `prefect deploy` CLI are then overlaid on the values loaded from the file.
  * The final step occurs when the fully realized deployment specification is registered with the Prefect API.


**Deployment instruction overrides** The `build`, `push`, and `pull` sections in deployment definitions take precedence over the corresponding sections above them in `prefect.yaml`.
Each time a step runs, the following actions take place in order:
  * The step’s inputs and block / variable references resolve.
  * The step’s function is imported; if it cannot be found, the special `requires` keyword installs the necessary packages.
  * The step’s function is called with the resolved inputs.
  * The step’s output is returned and used to resolve inputs for subsequent steps.


## 
[​](https://docs.prefect.io/v3/how-to-guides/deployments/prefect-yaml#update-a-deployment)
Update a deployment
To update a deployment, make any desired changes to the `prefect.yaml` file, and run `prefect deploy`. Running just this command will prompt you to select a deployment interactively, or you may specify the deployment to update with `--name your-deployment`.
## 
[​](https://docs.prefect.io/v3/how-to-guides/deployments/prefect-yaml#further-reading)
Further reading
Now that you are familiar with creating deployments, you can explore infrastructure options for running your deployments:
  * [Managed work pools](https://docs.prefect.io/v3/how-to-guides/deployment_infra/managed)
  * [Push work pools](https://docs.prefect.io/v3/how-to-guides/deployment_infra/serverless)
  * [Kubernetes work pools](https://docs.prefect.io/v3/how-to-guides/deployment_infra/kubernetes)


Was this page helpful?
YesNo
[Deploy via Python](https://docs.prefect.io/v3/how-to-guides/deployments/deploy-via-python)[Retrieve code from storage](https://docs.prefect.io/v3/how-to-guides/deployments/store-flow-code)
Deployments
# Work pools
Learn how to configure dynamic infrastructure provisioning with work pools
Work pools are a bridge between the Prefect orchestration layer and the infrastructure where flows are run. The primary reason to use work pools is for **dynamic infrastructure provisioning and configuration**. For example, you might have a workflow that has expensive infrastructure requirements and runs infrequently. In this case, you don’t want an idle process running within that infrastructure. Other advantages of work pools:
  * Configure default infrastructure configurations on your work pools that all jobs inherit and can override.
  * Allow platform teams to use work pools to expose opinionated (and enforced) interfaces to the infrastructure that they oversee.
  * Allow work pools to prioritize (or limit) flow runs through the use of [work queues](https://docs.prefect.io/v3/deploy/infrastructure-concepts/work-pools#work-queues).

Work pools remain a consistent interface for configuring deployment infrastructure, but only some work pool types require you to run a [worker](https://docs.prefect.io/v3/concepts/workers). Type | Description | You run a worker  
---|---|---  
[Hybrid](https://docs.prefect.io/v3/concepts/workers) | a worker in your infrastructure submits runs to your infrastructure | Yes  
[Push](https://docs.prefect.io/v3/how-to-guides/deployment_infra/serverless) | runs are automatically submitted to your configured serverless infrastructure provider | No  
[Managed](https://docs.prefect.io/v3/how-to-guides/deployment_infra/managed) | runs are automatically submitted to Prefect-managed infrastructure | No  
Each type of work pool is optimized for different use cases, allowing you to choose the best fit for your specific infrastructure and workflow requirements. By using work pools, you can efficiently manage the distribution and execution of your Prefect flows across environments and infrastructures.
**Work pools are like pub/sub topics** Work pools help coordinate deployments with workers through a known channel: the pool itself. This is similar to how “topics” are used to connect producers and consumers in a pub/sub or message-based system. By switching a deployment’s work pool, users can quickly change the worker that will execute their runs, making it easy to promote runs through environments — or even to debug locally.
The following diagram provides a high-level overview of the conceptual elements involved in defining a work-pool based deployment that is polled by a worker and executes a flow run based on that deployment.
Infrastructure
Remote Storage
Prefect API
Deployment Definition
Deployment
Flow Code
Worker
Flow Run
### 
[​](https://docs.prefect.io/v3/concepts/work-pools#work-pool-types)
Work pool types
The following work pool types are supported by Prefect:
  * Prefect Cloud
  * Self-hosted Prefect server


Infrastructure Type | Description  
---|---  
Process | Execute flow runs as subprocesses on a worker. Works well for local execution when first getting started.  
AWS Elastic Container Service | Execute flow runs within containers on AWS ECS. Works with EC2 and Fargate clusters. Requires an AWS account.  
Azure Container Instances | Execute flow runs within containers on Azure’s Container Instances service. Requires an Azure account.  
Docker | Execute flow runs within Docker containers. Works well for managing flow execution environments through Docker images. Requires access to a running Docker daemon.  
Google Cloud Run | Execute flow runs within containers on Google Cloud Run. Requires a Google Cloud Platform account.  
Google Cloud Run V2 | Execute flow runs within containers on Google Cloud Run (V2 API). Requires a Google Cloud Platform account.  
Google Vertex AI | Execute flow runs within containers on Google Vertex AI. Requires a Google Cloud Platform account.  
Kubernetes | Execute flow runs within jobs scheduled on a Kubernetes cluster. Requires a Kubernetes cluster.  
Google Cloud Run - Push | Execute flow runs within containers on Google Cloud Run. Requires a Google Cloud Platform account. Flow runs are pushed directly to your environment, without the need for a Prefect worker.  
AWS Elastic Container Service - Push | Execute flow runs within containers on AWS ECS. Works with existing ECS clusters and serverless execution through AWS Fargate. Requires an AWS account. Flow runs are pushed directly to your environment, without the need for a Prefect worker.  
Azure Container Instances - Push | Execute flow runs within containers on Azure’s Container Instances service. Requires an Azure account. Flow runs are pushed directly to your environment, without the need for a Prefect worker.  
Modal - Push | Execute [flow runs on Modal](https://docs.prefect.io/v3/how-to-guides/deployment_infra/modal). Requires a Modal account. Flow runs are pushed directly to your Modal workspace, without the need for a Prefect worker.  
Coiled | Execute flow runs in the cloud platform of your choice with Coiled. Makes it easy to run in your account without setting up Kubernetes or other cloud infrastructure.  
Prefect Managed | Execute flow runs within containers on Prefect managed infrastructure.  
### 
[​](https://docs.prefect.io/v3/concepts/work-pools#work-queues)
Work queues
Work queues offer advanced control over how runs are executed. Each work pool has a “default” queue which is used if another work queue name is not specified. Add additional queues to a work pool to enable greater control over work delivery through fine-grained priority and concurrency.
#### 
[​](https://docs.prefect.io/v3/concepts/work-pools#queue-priority)
Queue priority
Each work queue has a priority indicated by a unique positive integer. Lower numbers take greater priority in the allocation of work with `1` being the highest priority. You can add new queues without changing the rank of the higher-priority queues.
#### 
[​](https://docs.prefect.io/v3/concepts/work-pools#queue-concurrency-limits)
Queue concurrency limits
Work queues can also have their own concurrency limits. Each queue is also subject to the global work pool concurrency limit, which cannot be exceeded.
#### 
[​](https://docs.prefect.io/v3/concepts/work-pools#precise-control-with-priority-and-concurrency)
Precise control with priority and concurrency
Together, work queue priority and concurrency enable precise control over work. For example, a pool may have three queues:
  * a “low” queue with priority `10` and no concurrency limit
  * a “high” queue with priority `5` and a concurrency limit of `3`
  * a “critical” queue with priority `1` and a concurrency limit of `1`

This arrangement enables a pattern of two levels of priority: “high” and “low” for regularly scheduled flow runs, with the remaining “critical” queue for unplanned, urgent work, such as a backfill. Priority determines the order of flow runs submitted for execution. If all flow runs are capable of being executed with no limitation due to concurrency or otherwise, priority is still used to determine order of submission, but there is no impact to execution. If not all flow runs can execute, usually as a result of concurrency limits, priority determines which queues receive precedence to submit runs for execution. Priority for flow run submission proceeds from the highest priority to the lowest priority. In the previous example, all work from the “critical” queue (priority 1) is submitted, before any work is submitted from “high” (priority 5). Once all work is submitted from priority queue “critical”, work from the “high” queue begins submission. If new flow runs are received on the “critical” queue while flow runs are still in scheduled on the “high” and “low” queues, flow run submission goes back to ensuring all scheduled work is first satisfied. This happens from the highest priority queue, until it is empty, in waterfall fashion.
**Work queue status** A work queue has a `READY` status when it has been polled by a worker in the last 60 seconds. Pausing a work queue gives it a `PAUSED` status and means that it will accept no new work until it is unpaused. A user can control the work queue’s paused status in the UI. Unpausing a work queue gives the work queue a `NOT_READY` status unless a worker has polled it in the last 60 seconds.
## 
[​](https://docs.prefect.io/v3/concepts/work-pools#further-reading)
Further reading
  * Learn more about [workers](https://docs.prefect.io/v3/deploy/infrastructure-concepts/workers) and how they interact with work pools
  * Learn how to [deploy flows](https://docs.prefect.io/v3/deploy/infrastructure-concepts/prefect-yaml) that run in work pools
  * Learn how to set up work pools for:
    * [Kubernetes](https://docs.prefect.io/v3/how-to-guides/deployment_infra/kubernetes)
    * [Docker](https://docs.prefect.io/v3/how-to-guides/deployment_infra/docker)
    * [Serverless platforms](https://docs.prefect.io/v3/how-to-guides/deployment_infra/serverless)
    * [Infrastructure managed by Prefect Cloud](https://docs.prefect.io/v3/how-to-guides/deployment_infra/managed)


Was this page helpful?
YesNo
[Schedules](https://docs.prefect.io/v3/concepts/schedules)[Workers](https://docs.prefect.io/v3/concepts/workers)
Workflow Infrastructure
# How to run flows on serverless compute
Learn how to use Prefect push work pools to schedule work on serverless infrastructure without having to run a worker.
[work pools](https://docs.prefect.io/v3/deploy/infrastructure-concepts/work-pools) are a special type of work pool. They allow Prefect Cloud to submit flow runs for execution to serverless computing infrastructure without requiring you to run a worker. Push work pools currently support execution in AWS ECS tasks, Azure Container Instances, Google Cloud Run jobs, Modal, and Coiled. In this guide you will:
  * Create a push work pool that sends work to Amazon Elastic Container Service (AWS ECS), Azure Container Instances (ACI), Google Cloud Run, Modal, or Coiled
  * Deploy a flow to that work pool
  * Execute a flow without having to run a worker process to poll for flow runs


You can automatically provision infrastructure and create your push work pool using the `prefect work-pool create` CLI command with the `--provision-infra` flag. This approach greatly simplifies the setup process.
First, you will set up automatic infrastructure provisioning for push work pools. Then you will learn how to manually set up your push work pool.
## 
[​](https://docs.prefect.io/v3/how-to-guides/deployment_infra/serverless#automatic-infrastructure-provisioning)
Automatic infrastructure provisioning
With Perfect Cloud you can provision infrastructure for use with an AWS ECS, Google Cloud Run, ACI push work pool. Push work pools in Prefect Cloud simplify the setup and management of the infrastructure necessary to run your flows. However, setting up infrastructure on your cloud provider can still be a time-consuming process. Prefect dramatically simplifies this process by automatically provisioning the necessary infrastructure for you. We’ll use the `prefect work-pool create` CLI command with the `--provision-infra` flag to automatically provision your serverless cloud resources and set up your Prefect workspace to use a new push pool.
### 
[​](https://docs.prefect.io/v3/how-to-guides/deployment_infra/serverless#prerequisites)
Prerequisites
To use automatic infrastructure provisioning, you need:
  * the relevant cloud CLI library installed
  * to be authenticated with your cloud provider


  * AWS ECS
  * Azure Container Instances
  * Google Cloud Run
  * Modal
  * Coiled


Install the [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html), [authenticate with your AWS account](https://docs.aws.amazon.com/signin/latest/userguide/command-line-sign-in.html), and [set a default region](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html#cli-configure-files-methods).If you already have the AWS CLI installed, be sure to [update to the latest version](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html#getting-started-install-instructions).You will need the following permissions in your authenticated AWS account:IAM Permissions:
  * iam:CreatePolicy
  * iam:GetPolicy
  * iam:ListPolicies
  * iam:CreateUser
  * iam:GetUser
  * iam:AttachUserPolicy
  * iam:CreateRole
  * iam:GetRole
  * iam:AttachRolePolicy
  * iam:ListRoles
  * iam:PassRole

Amazon ECS Permissions:
  * ecs:CreateCluster
  * ecs:DescribeClusters

Amazon EC2 Permissions:
  * ec2:CreateVpc
  * ec2:DescribeVpcs
  * ec2:CreateInternetGateway
  * ec2:AttachInternetGateway
  * ec2:CreateRouteTable
  * ec2:CreateRoute
  * ec2:CreateSecurityGroup
  * ec2:DescribeSubnets
  * ec2:CreateSubnet
  * ec2:DescribeAvailabilityZones
  * ec2:AuthorizeSecurityGroupIngress
  * ec2:AuthorizeSecurityGroupEgress

Amazon ECR Permissions:
  * ecr:CreateRepository
  * ecr:DescribeRepositories
  * ecr:GetAuthorizationToken

If you want to use AWS managed policies, you can use the following:
  * AmazonECS_FullAccess
  * AmazonEC2FullAccess
  * IAMFullAccess
  * AmazonEC2ContainerRegistryFullAccess

The above policies give you all the permissions needed, but are more permissive than necessary.[Docker](https://docs.docker.com/get-docker/) is also required to build and push images to your registry.
### 
[​](https://docs.prefect.io/v3/how-to-guides/deployment_infra/serverless#automatically-create-a-new-push-work-pool-and-provision-infrastructure)
Automatically create a new push work pool and provision infrastructure
To create a new push work pool and configure the necessary infrastructure, run this command for your particular cloud provider:
  * AWS ECS
  * Azure Container Instances
  * Google Cloud Run
  * Modal
  * Coiled


Copy
```
prefect work-pool create --type ecs:push --provision-infra my-ecs-pool

```

The `--provision-infra` flag automatically sets up your default AWS account to execute flows with ECS tasks. In your AWS account, this command creates a new IAM user, IAM policy, and ECS cluster that uses AWS Fargate, VPC, and ECR repository (if they don’t already exist). In your Prefect workspace, this command creates an [`AWSCredentials` block](https://docs.prefect.io/integrations/prefect-aws/index#save-credentials-to-an-aws-credentials-block) for storing the generated credentials.Here’s an abbreviated example output from running the command:
Copy
```
_____________________________________________________________________________________________
| Provisioning infrastructure for your work pool my-ecs-pool will require:                   |
|                                                                                            |
|          - Creating an IAM user for managing ECS tasks: prefect-ecs-user                   |
|          - Creating and attaching an IAM policy for managing ECS tasks: prefect-ecs-policy |
|          - Storing generated AWS credentials in a block                                    |
|          - Creating an ECS cluster for running Prefect flows: prefect-ecs-cluster          |
|          - Creating a VPC with CIDR 172.31.0.0/16 for running ECS tasks: prefect-ecs-vpc   |
|          - Creating an ECR repository for storing Prefect images: prefect-flows            |
_____________________________________________________________________________________________
Proceed with infrastructure provisioning? [y/n]: y
Provisioning IAM user
Creating IAM policy
Generating AWS credentials
Creating AWS credentials block
Provisioning ECS cluster
Provisioning VPC
Creating internet gateway
Setting up subnets
Setting up security group
Provisioning ECR repository
Authenticating with ECR
Setting default Docker build namespace
Provisioning Infrastructure ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
Infrastructure successfully provisioned!
Created work pool 'my-ecs-pool'!

```

**Default Docker build namespace** After infrastructure provisioning completes, you will be logged into your new ECR repository and the default Docker build namespace will be set to the URL of the registry.
While the default namespace is set, you do not need to provide the registry URL when building images as part of your deployment process.To take advantage of this, you can write your deploy scripts like this:
example_deploy_script.py
Copy
```
from prefect import flow
from prefect.docker import DockerImage
@flow(log_prints=True)
def my_flow(name: str = "world"):
    print(f"Hello {name}! I'm a flow running in a ECS task!")
if __name__ == "__main__":
    my_flow.deploy(
        name="my-deployment",
        work_pool_name="my-work-pool",
        image=DockerImage(
            name="my-repository:latest",
            platform="linux/amd64",
        )
    )

```

This builds an image with the tag `<ecr-registry-url>/my-image:latest` and push it to the registry.Your image name needs to match the name of the repository created with your work pool. You can create new repositories in the ECR console.
You’re ready to create and schedule deployments that use your new push work pool. Reminder that no worker is required to run flows with a push work pool.
### 
[​](https://docs.prefect.io/v3/how-to-guides/deployment_infra/serverless#use-existing-resources-with-automatic-infrastructure-provisioning)
Use existing resources with automatic infrastructure provisioning
If you already have the necessary infrastructure set up, Prefect detects that at work pool creation and the infrastructure provisioning for that resource is skipped. For example, here’s how `prefect work-pool create my-work-pool --provision-infra` looks when existing Azure resources are detected:
Copy
```
Proceed with infrastructure provisioning? [y/n]: y
Creating resource group
Resource group 'prefect-aci-push-pool-rg' already exists in location 'eastus'.
Creating app registration
App registration 'prefect-aci-push-pool-app' already exists.
Generating secret for app registration
Provisioning infrastructure
ACI credentials block 'bb-push-pool-credentials' created
Assigning Contributor role to service account...
Service principal with object ID '4be6fed7-...' already has the 'Contributor' role assigned in
'/subscriptions/.../'
Creating Azure Container Instance
Container instance 'prefect-aci-push-pool-container' already exists.
Creating Azure Container Instance credentials block
Provisioning infrastructure... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
Infrastructure successfully provisioned!
Created work pool 'my-work-pool'!

```

## 
[​](https://docs.prefect.io/v3/how-to-guides/deployment_infra/serverless#provision-infrastructure-for-an-existing-push-work-pool)
Provision infrastructure for an existing push work pool
If you already have a push work pool set up, but haven’t configured the necessary infrastructure, you can use the `provision-infra` sub-command to provision the infrastructure for that work pool. For example, you can run the following command if you have a work pool named “my-work-pool”.
Copy
```
prefect work-pool provision-infra my-work-pool

```

Prefect creates the necessary infrastructure for the `my-work-pool` work pool and provides you with a summary of the changes:
Copy
```
__________________________________________________________________________________________________________________
| Provisioning infrastructure for your work pool my-work-pool will require:                                      |
|                                                                                                                |
|     Updates in subscription Azure subscription 1                                                               |
|                                                                                                                |
|         - Create a resource group in location eastus                                                           |
|         - Create an app registration in Azure AD prefect-aci-push-pool-app                                     |
|         - Create/use a service principal for app registration                                                  |
|         - Generate a secret for app registration                                                               |
|         - Assign Contributor role to service account                                                           |
|         - Create Azure Container Instance 'aci-push-pool-container' in resource group prefect-aci-push-pool-rg |
|                                                                                                                |
|     Updates in Prefect workspace                                                                               |
|                                                                                                                |
|         - Create Azure Container Instance credentials block aci-push-pool-credentials                          |
|                                                                                                                |
__________________________________________________________________________________________________________________
Proceed with infrastructure provisioning? [y/n]: y

```

This command speeds up your infrastructure setup process. As with the examples above, you need to have the related cloud CLI library installed and to be authenticated with your cloud provider.
## 
[​](https://docs.prefect.io/v3/how-to-guides/deployment_infra/serverless#manual-infrastructure-provisioning)
Manual infrastructure provisioning
If you prefer to set up your infrastructure manually, exclude the `--provision-infra` flag in the CLI command. In the examples below, you’ll create a push work pool through the Prefect Cloud UI.
  * AWS ECS
  * Azure Container Instances
  * Google Cloud Run
  * Modal
  * Coiled


To push work to ECS, AWS credentials are required.Create a user and attach the _AmazonECS_FullAccess_ permissions.From that user’s page, create credentials and store them somewhere safe for use in the next section.
### 
[​](https://docs.prefect.io/v3/how-to-guides/deployment_infra/serverless#work-pool-configuration)
Work pool configuration
The push work pool stores information about what type of infrastructure the flow will run on, what default values to provide to compute jobs, and other important execution environment parameters. Because the push work pool needs to integrate securely with your serverless infrastructure, you need to store your credentials in Prefect Cloud by making a block.
### 
[​](https://docs.prefect.io/v3/how-to-guides/deployment_infra/serverless#create-a-credentials-block)
Create a Credentials block
  * AWS ECS
  * Azure Container Instances
  * Google Cloud Run
  * Modal
  * Coiled


Navigate to the blocks page, click create new block, and select AWS Credentials for the type.For use in a push work pool, set the region, access key, and access key secret.Provide any other optional information and create your block.
### 
[​](https://docs.prefect.io/v3/how-to-guides/deployment_infra/serverless#create-a-push-work-pool)
Create a push work pool
Now navigate to the work pools page. Click **Create** to configure your push work pool by selecting a push option in the infrastructure type step.
  * AWS ECS
  * Azure Container Instances
  * Google Cloud Run
  * Modal
  * Coiled


Each step has several optional fields that are detailed in the [work pools documentation](https://docs.prefect.io/v3/deploy/infrastructure-concepts/work-pools). Select the block you created under the AWS Credentials field. This allows Prefect Cloud to securely interact with your ECS cluster.
Create your pool to be ready to deploy flows to your Push work pool.
## 
[​](https://docs.prefect.io/v3/how-to-guides/deployment_infra/serverless#deployment)
Deployment
You need to configure your [deployment](https://docs.prefect.io/v3/how-to-guides/deployment_infra/docker) to send flow runs to your push work pool. For example, if you create a deployment through the interactive command line experience, choose the work pool you just created. If you are deploying an existing `prefect.yaml` file, the deployment would contain:
Copy
```
  work_pool:
    name: my-push-pool

```

Deploying your flow to the `my-push-pool` work pool ensures that runs that are ready for execution are submitted immediately—without the need for a worker to poll for them.
**Serverless infrastructure may require a certain image architecture** Serverless infrastructure may assume a certain Docker image architecture; for example, Google Cloud Run will fail to run images built with `linux/arm64` architecture. If using Prefect to build your image, you can change the image architecture through the `platform` keyword (for example, `platform="linux/amd64"`).
## 
[​](https://docs.prefect.io/v3/how-to-guides/deployment_infra/serverless#putting-it-all-together)
Putting it all together
With your deployment created, navigate to its detail page and create a new flow run. You’ll see the flow start running without polling the work pool, because Prefect Cloud securely connected to your serverless infrastructure, created a job, ran the job, and reported on its execution.
## 
[​](https://docs.prefect.io/v3/how-to-guides/deployment_infra/serverless#usage-limits)
Usage Limits
Push work pool usage is unlimited. However push work pools limit flow runs to 24 hours.
## 
[​](https://docs.prefect.io/v3/how-to-guides/deployment_infra/serverless#next-steps)
Next steps
Learn more about [work pools](https://docs.prefect.io/v3/deploy/infrastructure-concepts/work-pools) and [workers](https://docs.prefect.io/v3/deploy/infrastructure-concepts/workers). Learn about installing dependencies at runtime or baking them into your Docker image in the [Deploy to Docker](https://docs.prefect.io/v3/how-to-guides/deployment_infra/docker#automatically-build-a-custom-docker-image-with-a-local-dockerfile) guide.
Was this page helpful?
YesNo
[Run flows on Prefect Managed infrastructure](https://docs.prefect.io/v3/how-to-guides/deployment_infra/managed)[Run flows in Docker containers](https://docs.prefect.io/v3/how-to-guides/deployment_infra/docker)
Workflow Infrastructure
# How to run flows in local processes
Create a deployment for a flow by calling the `serve` method.
The simplest way to create a [deployment](https://docs.prefect.io/v3/deploy) for your flow is by calling its `serve` method.
## 
[​](https://docs.prefect.io/v3/how-to-guides/deployment_infra/run-flows-in-local-processes#serve-a-flow)
Serve a flow
The serve method creates a deployment for the flow and starts a long-running process that monitors for work from the Prefect server. When work is found, it is executed within its own isolated subprocess.
hello_world.py
Copy
```
from prefect import flow
@flow(log_prints=True)
def hello_world(name: str = "world", goodbye: bool = False):
    print(f"Hello {name} from Prefect! 🤗")
    if goodbye:
        print(f"Goodbye {name}!")
if __name__ == "__main__":
    # creates a deployment and starts a long-running
    # process that listens for scheduled work
    hello_world.serve(name="my-first-deployment",
        tags=["onboarding"],
        parameters={"goodbye": True},
        interval=60
    )

```

This interface provides the configuration for a deployment (with no strong infrastructure requirements), such as:
  * schedules
  * event triggers
  * metadata such as tags and description
  * default parameter values


**Schedules are auto-paused on shutdown** By default, stopping the process running `flow.serve` will pause the schedule for the deployment (if it has one).When running this in environments where restarts are expected use the`pause_on_shutdown=False` flag to prevent this behavior:
Copy
```
if __name__ == "__main__":
    hello_world.serve(
        name="my-first-deployment",
        tags=["onboarding"],
        parameters={"goodbye": True},
        pause_on_shutdown=False,
        interval=60
    )

```

## 
[​](https://docs.prefect.io/v3/how-to-guides/deployment_infra/run-flows-in-local-processes#additional-serve-options)
Additional serve options
The `serve` method on flows exposes many options for the deployment. Here’s how to use some of those options:
  * `cron`: a keyword that allows you to set a cron string schedule for the deployment; see [schedules](https://docs.prefect.io/v3/automate/add-schedules) for more advanced scheduling options
  * `tags`: a keyword that allows you to tag this deployment and its runs for bookkeeping and filtering purposes
  * `description`: a keyword that allows you to document what this deployment does; by default the description is set from the docstring of the flow function (if documented)
  * `version`: a keyword that allows you to track changes to your deployment; uses a hash of the file containing the flow by default; popular options include semver tags or git commit hashes
  * `triggers`: a keyword that allows you to define a set of conditions for when the deployment should run; see [triggers](https://docs.prefect.io/v3/concepts/event-triggers) for more on Prefect Events concepts

Next, add these options to your deployment:
Copy
```
if __name__ == "__main__":
    get_repo_info.serve(
        name="my-first-deployment",
        cron="* * * * *",
        tags=["testing", "tutorial"],
        description="Given a GitHub repository, logs repository statistics for that repo.",
        version="tutorial/deployments",
    )

```

**Triggers with`.serve`** See this [example](https://docs.prefect.io/v3/how-to-guides/automations/chaining-deployments-with-events) that triggers downstream work on upstream events.
When you rerun this script, you will find an updated deployment in the UI that is actively scheduling work. Stop the script in the CLI using `CTRL+C` and your schedule automatically pauses.
**`serve()`is a long-running process** To execute remotely triggered or scheduled runs, your script with `flow.serve` must be actively running.
## 
[​](https://docs.prefect.io/v3/how-to-guides/deployment_infra/run-flows-in-local-processes#serve-multiple-flows-at-once)
Serve multiple flows at once
Serve multiple flows with the same process using the `serve` utility along with the `to_deployment` method of flows:
serve_two_flows.py
Copy
```
import time
from prefect import flow, serve
@flow
def slow_flow(sleep: int = 60):
    "Sleepy flow - sleeps the provided amount of time (in seconds)."
    time.sleep(sleep)
@flow
def fast_flow():
    "Fastest flow this side of the Mississippi."
    return
if __name__ == "__main__":
    slow_deploy = slow_flow.to_deployment(name="sleeper", interval=45)
    fast_deploy = fast_flow.to_deployment(name="fast")
    serve(slow_deploy, fast_deploy)

```

The behavior and interfaces are identical to the single flow case. A few things to note:
  * the `flow.to_deployment` interface exposes the _exact same_ options as `flow.serve`; this method produces a deployment object
  * the deployments are only registered with the API once `serve(...)` is called
  * when serving multiple deployments, the only requirement is that they share a Python environment; they can be executed and scheduled independently of each other

A few optional steps for exploration include:
  * pause and unpause the schedule for the `"sleeper"` deployment
  * use the UI to submit ad-hoc runs for the `"sleeper"` deployment with different values for `sleep`
  * cancel an active run for the `"sleeper"` deployment from the UI


**Hybrid execution option** Prefect’s deployment interface allows you to choose a hybrid execution model. Whether you use Prefect Cloud or self-host Prefect server, you can run workflows in the environments best suited to their execution. This model enables efficient use of your infrastructure resources while maintaining the privacy of your code and data. There is no ingress required. Read more about our [hybrid model](https://www.prefect.io/security/overview/#hybrid-model).
## 
[​](https://docs.prefect.io/v3/how-to-guides/deployment_infra/run-flows-in-local-processes#retrieve-a-flow-from-remote-storage)
Retrieve a flow from remote storage
Just like the `.deploy` method, the `flow.from_source` method is used to define how to retrieve the flow that you want to serve.
### 
[​](https://docs.prefect.io/v3/how-to-guides/deployment_infra/run-flows-in-local-processes#from-source)
`from_source`
The `flow.from_source` method on `Flow` objects requires a `source` and an `entrypoint`.
#### 
[​](https://docs.prefect.io/v3/how-to-guides/deployment_infra/run-flows-in-local-processes#source)
`source`
The `source` of your deployment can be:
  * a path to a local directory such as `path/to/a/local/directory`
  * a repository URL such as `https://github.com/org/repo.git`
  * a `GitRepository` object that accepts 
    * a repository URL
    * a reference to a branch, tag, or commit hash
    * `GitCredentials` for private repositories


#### 
[​](https://docs.prefect.io/v3/how-to-guides/deployment_infra/run-flows-in-local-processes#entrypoint)
`entrypoint`
A flow `entrypoint` is the path to the file where the flow is located within that `source`, in the form
Copy
```
{path}:{flow_name}

```

For example, the following code will load the `hello` flow from the `flows/hello_world.py` file in the `PrefectHQ/examples` repository:
load_from_url.py
Copy
```
from prefect import flow
my_flow = flow.from_source(
    source="https://github.com/PrefectHQ/examples.git",
    entrypoint="flows/hello_world.py:hello"
)
if __name__ == "__main__":
    my_flow()

```

Copy
```
16:40:33.818 | INFO    | prefect.engine - Created flow run 'muscular-perch' for flow 'hello'
16:40:34.048 | INFO    | Flow run 'muscular-perch' - Hello world!
16:40:34.706 | INFO    | Flow run 'muscular-perch' - Finished in state Completed()

```

For more ways to store and access flow code, see the [Retrieve code from storage page](https://docs.prefect.io/v3/deploy/infrastructure-concepts/store-flow-code).
**You can serve loaded flows** You can serve a flow loaded from remote storage with the same [`serve`](https://docs.prefect.io/v3/how-to-guides/deployment_infra/run-flows-in-local-processes#serve-a-flow) method as a local flow:
serve_loaded_flow.py
Copy
```
from prefect import flow
if __name__ == "__main__":
    flow.from_source(
        source="https://github.com/org/repo.git",
        entrypoint="flows.py:my_flow"
    ).serve(name="my-deployment")

```

### 
[​](https://docs.prefect.io/v3/how-to-guides/deployment_infra/run-flows-in-local-processes#remote-storage-polling)
Remote storage polling
When you serve a flow loaded from remote storage, the serving process periodically polls your remote storage for updates to the flow’s code. This pattern allows you to update your flow code without restarting the serving process. Note that if you change metadata associated with your flow’s deployment such as parameters, you _will_ need to restart the serve process.
## 
[​](https://docs.prefect.io/v3/how-to-guides/deployment_infra/run-flows-in-local-processes#further-reading)
Further reading
  * [Serve flows in a long-lived Docker container](https://docs.prefect.io/v3/deploy/static-infrastructure-examples/docker)
  * [Work pools and deployments with dynamic infrastructure](https://docs.prefect.io/v3/deploy/infrastructure-concepts/work-pools)


Was this page helpful?
YesNo
[Manage Work Pools](https://docs.prefect.io/v3/how-to-guides/deployment_infra/manage-work-pools)[Run flows on Prefect Managed infrastructure](https://docs.prefect.io/v3/how-to-guides/deployment_infra/managed)
Workflow Infrastructure
# How to run flows in Docker containers
Learn how to execute deployments in isolated Docker containers
In this example, you will set up:
  * a Docker [**work pool**](https://docs.prefect.io/v3/deploy/infrastructure-concepts/work-pools): stores the infrastructure configuration for your deployment
  * a Docker [**worker**](https://docs.prefect.io/v3/deploy/infrastructure-concepts/workers): process that polls the Prefect API for flow runs to execute as Docker containers
  * a [**deployment**](https://docs.prefect.io/v3/deploy/index): a flow that should run according to the configuration on your Docker work pool

Then you can execute your deployment via the Prefect API (through the SDK, CLI, UI, etc). You must have [Docker](https://docs.docker.com/engine/install/) installed and running on your machine.
**Executing flows in a long-lived container** This guide shows how to run a flow in an ephemeral container that is removed after the flow run completes. To instead learn how to run flows in a static, long-lived container, see [this](https://docs.prefect.io/v3/deploy/static-infrastructure-examples/docker) guide.
### 
[​](https://docs.prefect.io/v3/how-to-guides/deployment_infra/docker#create-a-work-pool)
Create a work pool
A work pool provides default infrastructure configurations that all jobs inherit and can override. You can adjust many defaults, such as the base Docker image, container cleanup behavior, and resource limits. To set up a **Docker** type work pool with the default values, run:
Copy
```
prefect work-pool create --type docker my-docker-pool

```

… or create the work pool in the UI. To confirm the work pool creation was successful, run:
Copy
```
prefect work-pool ls

```

You should see your new `my-docker-pool` listed in the output. Next, check that you can see this work pool in your Prefect UI. Navigate to the **Work Pools** tab and verify that you see `my-docker-pool` listed. When you click into `my-docker-pool`, you should see a red status icon signifying that this work pool is not ready. To make the work pool ready, you’ll need to start a worker. We’ll show how to do this next.
### 
[​](https://docs.prefect.io/v3/how-to-guides/deployment_infra/docker#start-a-worker)
Start a worker
Workers are a lightweight polling process that kick off scheduled flow runs on a specific type of infrastructure (such as Docker). To start a worker on your local machine, open a new terminal and confirm that your virtual environment has `prefect` installed. Run the following command in this new terminal to start the worker:
Copy
```
prefect worker start --pool my-docker-pool

```

You should see the worker start. It’s now polling the Prefect API to check for any scheduled flow runs it should pick up and then submit for execution. You’ll see your new worker listed in the UI under the **Workers** tab of the Work Pools page with a recent last polled date. The work pool should have a `Ready` status indicator.
**Pro Tip:** If `my-docker-pool` does not already exist, the below command will create it for you automatically with the default settings for that work pool type, in this case `docker`.
Copy
```
prefect worker start --pool my-docker-pool --type docker

```

Keep this terminal session active for the worker to continue to pick up jobs. Since you are running this worker locally, the worker will if you close the terminal. In a production setting this worker should run as a [daemonized or managed process](https://docs.prefect.io/v3/deploy/daemonize-processes).
## 
[​](https://docs.prefect.io/v3/how-to-guides/deployment_infra/docker#create-the-deployment)
Create the deployment
From the previous steps, you now have:
  * A work pool
  * A worker

Next, you’ll create a deployment from your flow code.
### 
[​](https://docs.prefect.io/v3/how-to-guides/deployment_infra/docker#automatically-bake-your-code-into-a-docker-image)
Automatically bake your code into a Docker image
Create a deployment from Python code by calling the `.deploy` method on a flow:
deploy_buy.py
Copy
```
from prefect import flow
@flow(log_prints=True)
def buy():
    print("Buying securities")
if __name__ == "__main__":
    buy.deploy(
        name="my-code-baked-into-an-image-deployment",
        work_pool_name="my-docker-pool",
        image="my_registry/my_image:my_image_tag" # YOUR IMAGE REGISTRY
    )

```

Now, run the script to create a deployment (in future examples this step is omitted for brevity):
Copy
```
python deploy_buy.py

```

You should see messages in your terminal that Docker is building your image. When the deployment build succeeds, you will see information in your terminal showing you how to start a worker for your deployment, and how to run your deployment. Your deployment is visible on the `Deployments` page in the UI. By default, `.deploy` builds a Docker image with your flow code baked into it and pushes the image to the [Docker Hub](https://hub.docker.com/) registry implied by the `image` argument to `.deploy`.
**Authentication to Docker Hub** Your environment must be authenticated to your Docker registry to push an image to it.
You can specify a registry other than Docker Hub by providing the full registry path in the `image` argument.
If building a Docker image, your environment with your deployment needs Docker installed and running.
To avoid pushing to a registry, set `push=False` in the `.deploy` method:
Copy
```
if __name__ == "__main__":
    buy.deploy(
        name="my-code-baked-into-an-image-deployment",
        work_pool_name="my-docker-pool",
        image="my_registry/my_image:my_image_tag",
        push=False
    )

```

To avoid building an image, set `build=False` in the `.deploy` method:
Copy
```
if __name__ == "__main__":
    buy.deploy(
        name="my-code-baked-into-an-image-deployment",
        work_pool_name="my-docker-pool",
        image="my_registry/already-built-image:1.0",
        build=False
    )

```

The specified image must be available in your deployment’s execution environment for accessible flow code. Prefect generates a Dockerfile for you that builds an image based off of one of Prefect’s published images. The generated Dockerfile copies the current directory into the Docker image and installs any dependencies listed in a `requirements.txt` file.
### 
[​](https://docs.prefect.io/v3/how-to-guides/deployment_infra/docker#automatically-build-a-custom-docker-image-with-a-local-dockerfile)
Automatically build a custom Docker image with a local Dockerfile
If you want to use a custom image, specify the path to your Dockerfile via `DockerImage`:
my_flow.py
Copy
```
from prefect import flow
from prefect.docker import DockerImage
@flow(log_prints=True)
def buy():
    print("Buying securities")
if __name__ == "__main__":
    buy.deploy(
        name="my-custom-dockerfile-deployment",
        work_pool_name="my-docker-pool",
        image=DockerImage(
            name="my_image",
            tag="deploy-guide",
            dockerfile="Dockerfile"
    ),
    push=False
)

```

The `DockerImage` object enables image customization. For example, you can install a private Python package from GCP’s artifact registry like this:
  1. Create a custom base Dockerfile.
sample.Dockerfile
Copy
```
FROM python:3.12
ARG AUTHED_ARTIFACT_REG_URL
COPY ./requirements.txt /requirements.txt
RUN pip install --extra-index-url ${AUTHED_ARTIFACT_REG_URL} -r /requirements.txt

```

  2. Create your deployment with the `DockerImage` class:
deploy_using_private_package.py
Copy
```
from prefect import flow
from prefect.deployments.runner import DockerImage
from prefect.blocks.system import Secret
from myproject.cool import do_something_cool
@flow(log_prints=True)
def my_flow():
    do_something_cool()
if __name__ == "__main__":
    artifact_reg_url = Secret.load("artifact-reg-url")
    my_flow.deploy(
        name="my-deployment",
        work_pool_name="my-docker-pool",
        image=DockerImage(
            name="my-image",
            tag="test",
            dockerfile="sample.Dockerfile",
            buildargs={"AUTHED_ARTIFACT_REG_URL": artifact_reg_url.get()},
        ),
    )

```


Note that this example used a [Prefect Secret block](https://docs.prefect.io/v3/develop/blocks) to load the URL configuration for the artifact registry above. See all the optional keyword arguments for the [`DockerImage` class](https://docker-py.readthedocs.io/en/stable/images.html#docker.models.images.ImageCollection.build).
**Default Docker namespace** You can set the `PREFECT_DEFAULT_DOCKER_BUILD_NAMESPACE` setting to append a default Docker namespace to all images you build with `.deploy`. This is helpful if you use a private registry to store your images.To set a default Docker namespace for your current profile run:
Copy
```
prefect config set PREFECT_DEFAULT_DOCKER_BUILD_NAMESPACE=<docker-registry-url>/<organization-or-username>

```

Once set, you can omit the namespace from your image name when creating a deployment:
with_default_docker_namespace.py
Copy
```
if __name__ == "__main__":
    buy.deploy(
        name="my-code-baked-into-an-image-deployment",
        work_pool_name="my-docker-pool",
        image="my_image:my_image_tag"
    )

```

The above code builds an image with the format `<docker-registry-url>/<organization-or-username>/my_image:my_image_tag` when `PREFECT_DEFAULT_DOCKER_BUILD_NAMESPACE` is set.
### 
[​](https://docs.prefect.io/v3/how-to-guides/deployment_infra/docker#store-your-code-in-git-based-cloud-storage)
Store your code in git-based cloud storage
While baking code into Docker images is a popular deployment option, many teams store their workflow code in git-based storage, such as GitHub, Bitbucket, or GitLab. If you don’t specify an `image` argument for `.deploy`, you must specify where to pull the flow code from at runtime with the `from_source` method. Here’s how to pull your flow code from a GitHub repository:
git_storage.py
Copy
```
from prefect import flow
if __name__ == "__main__":
    flow.from_source(
        "https://github.com/my_github_account/my_repo/my_file.git",
        entrypoint="flows/no-image.py:hello_world",
    ).deploy(
        name="no-image-deployment",
        work_pool_name="my-docker-pool",
        build=False
    )

```

The `entrypoint` is the path to the file the flow is located in and the function name, separated by a colon. See the [Store flow code](https://docs.prefect.io/v3/deploy/infrastructure-concepts/store-flow-code) guide for more flow code storage options.
### 
[​](https://docs.prefect.io/v3/how-to-guides/deployment_infra/docker#additional-configuration-with-deploy)
Additional configuration with `.deploy`
Next, see deployment configuration options. To pass parameters to your flow, you can use the `parameters` argument in the `.deploy` method. Just pass in a dictionary of key-value pairs.
pass_params.py
Copy
```
from prefect import flow
@flow
def hello_world(name: str):
    print(f"Hello, {name}!")
if __name__ == "__main__":
    hello_world.deploy(
        name="pass-params-deployment",
        work_pool_name="my-docker-pool",
        parameters=dict(name="Prefect"),
        image="my_registry/my_image:my_image_tag",
    )

```

The `job_variables` parameter allows you to fine-tune the infrastructure settings for a deployment. The values passed in override default values in the specified work pool’s [base job template](https://docs.prefect.io/v3/deploy/infrastructure-concepts/work-pools#base-job-template). You can override environment variables, such as `image_pull_policy` and `image`, for a specific deployment with the `job_variables` argument.
job_var_image_pull.py
Copy
```
if __name__ == "__main__":
    get_repo_info.deploy(
        name="my-deployment-never-pull",
        work_pool_name="my-docker-pool",
        job_variables={"image_pull_policy": "Never"},
        image="my-image:my-tag",
        push=False
    )

```

Similarly, you can override the environment variables specified in a work pool through the `job_variables` parameter:
job_var_env_vars.py
Copy
```
if __name__ == "__main__":
    get_repo_info.deploy(
        name="my-deployment-never-pull",
        work_pool_name="my-docker-pool",
        job_variables={"env": {"EXTRA_PIP_PACKAGES": "boto3"} },
        image="my-image:my-tag",
        push=False
    )

```

The dictionary key “EXTRA_PIP_PACKAGES” denotes a special environment variable that Prefect uses to install additional Python packages at runtime. This approach is an alternative to building an image with a custom `requirements.txt` copied into it. See [Override work pool job variables](https://docs.prefect.io/v3/deploy/infrastructure-concepts/customize) for more information about how to customize these variables.
### 
[​](https://docs.prefect.io/v3/how-to-guides/deployment_infra/docker#work-with-multiple-deployments-with-deploy)
Work with multiple deployments with `deploy`
Create multiple deployments from one or more Python files that use `.deploy`. You can manage these deployments independently of one another to deploy the same flow with different configurations in the same codebase. To create multiple deployments at once, use the `deploy` function, which is analogous to the `serve` function:
Copy
```
from prefect import deploy, flow
@flow(log_prints=True)
def buy():
    print("Buying securities")
if __name__ == "__main__":
    deploy(
        buy.to_deployment(name="dev-deploy", work_pool_name="my-docker-pool"),
        buy.to_deployment(name="prod-deploy", work_pool_name="my-other-docker-pool"),
        image="my-registry/my-image:dev",
        push=False,
    )

```

In the example above you created two deployments from the same flow, but with different work pools. Alternatively, you can create two deployments from different flows:
Copy
```
from prefect import deploy, flow
@flow(log_prints=True)
def buy():
    print("Buying securities.")
@flow(log_prints=True)
def sell():
    print("Selling securities.")
if __name__ == "__main__":
    deploy(
        buy.to_deployment(name="buy-deploy"),
        sell.to_deployment(name="sell-deploy"),
        work_pool_name="my-docker-pool",
        image="my-registry/my-image:dev",
        push=False,
    )

```

In the example above, the code for both flows is baked into the same image. You can specify one or more flows to pull from a remote location at runtime with the `from_source` method. Here’s an example of deploying two flows, one defined locally and one defined in a remote repository:
Copy
```
from prefect import deploy, flow
@flow(log_prints=True)
def local_flow():
    print("I'm a flow!")
if __name__ == "__main__":
    deploy(
        local_flow.to_deployment(name="example-deploy-local-flow"),
        flow.from_source(
            source="https://github.com/org/repo.git",
            entrypoint="flows.py:my_flow",
        ).to_deployment(
            name="example-deploy-remote-flow",
        ),
        work_pool_name="my-docker-pool",
        image="my-registry/my-image:dev",
    )

```

You can pass any number of flows to the `deploy` function. This is useful if using a monorepo approach to your workflows.
## 
[​](https://docs.prefect.io/v3/how-to-guides/deployment_infra/docker#learn-more)
Learn more
  * [Deploy flows on Kubernetes](https://docs.prefect.io/v3/how-to-guides/deployment_infra/kubernetes)
  * [Deploy flows on serverless infrastructure](https://docs.prefect.io/v3/how-to-guides/deployment_infra/serverless)
  * [Daemonize workers](https://docs.prefect.io/v3/deploy/daemonize-processes)


Was this page helpful?
YesNo
[Run flows on serverless compute](https://docs.prefect.io/v3/how-to-guides/deployment_infra/serverless)[Run flows in a static container](https://docs.prefect.io/v3/how-to-guides/deployment_infra/serve-flows-docker)
Deployments
# Trigger ad-hoc deployment runs
Learn how to trigger deployment runs using the Prefect CLI and Python SDK.
[Deployments](https://docs.prefect.io/v3/concepts/deployments) are server-side representations of flows that can be executed:
  * on a [schedule](https://docs.prefect.io/v3/how-to-guides/deployments/create-schedules)
  * when [triggered by events](https://docs.prefect.io/v3/how-to-guides/automations/creating-deployment-triggers)
  * programmatically, on demand

This guide covers how to trigger deployments on demand.
## 
[​](https://docs.prefect.io/v3/how-to-guides/deployments/run-deployments#prerequisites)
Prerequisites
In order to run a deployment, you need to have:
  * [created a deployment](https://docs.prefect.io/v3/how-to-guides/deployments/create-deployments)
  * started a process ([`serve`](https://docs.prefect.io/v3/how-to-guides/deployment_infra/run-flows-in-local-processes) or a [worker](https://docs.prefect.io/v3/concepts/workers)) listening for scheduled runs of that deployment


## 
[​](https://docs.prefect.io/v3/how-to-guides/deployments/run-deployments#run-a-deployment-from-the-cli)
Run a deployment from the CLI
The simplest way to trigger a deployment run is using the Prefect CLI:
Copy
```
prefect deployment run my-flow/my-deployment

```

### 
[​](https://docs.prefect.io/v3/how-to-guides/deployments/run-deployments#cli-options)
CLI options
Add parameters and customize the run, including setting a custom flow run name using the new —flow-run-name option:
Copy
```
# Pass parameters
prefect deployment run my-flow/my-deployment \
  --param my_param=42 \
  --param another_param="hello"
# Schedule for later
prefect deployment run my-flow/my-deployment --start-in "2 hours"
# Watch the run until completion
prefect deployment run my-flow/my-deployment --watch
# Set custom run name
Use `--flow-run-name` to set a static or templated name for the flow run.
prefect deployment run my-flow/my-deployment --flow-run-name "custom-run-name"
# Set a custom flow run name using templating
prefect deployment run my-flow/my-deployment \
  --param customer_id=1234 \
  --param run_date="2025-07-14" \
  --flow-run-name "customer-{customer_id}-run-{run_date}"
> Note: You can use `{parameter}` syntax to template the flow run name.
> The values will be substituted from the `--param` values.
# Add tags to the run
prefect deployment run my-flow/my-deployment --tag production --tag critical

```

## 
[​](https://docs.prefect.io/v3/how-to-guides/deployments/run-deployments#run-a-deployment-from-python)
Run a deployment from Python
Use the `run_deployment` function for programmatic control:
Copy
```
from prefect.deployments import run_deployment
# Basic usage
flow_run = run_deployment(
    name="my-flow/my-deployment"
)
# With parameters
flow_run = run_deployment(
    name="my-flow/my-deployment",
    parameters={
        "my_param": 42,
        "another_param": "hello"
    }
)
# With job variables (environment variables, etc.)
flow_run = run_deployment(
    name="my-flow/my-deployment",
    parameters={"my_param": 42},
    job_variables={"env": {"MY_ENV_VAR": "production"}}
)
# Don't wait for completion
flow_run = run_deployment(
    name="my-flow/my-deployment",
    timeout=0  # returns immediately
)
# Wait with custom timeout (seconds)
flow_run = run_deployment(
    name="my-flow/my-deployment",
    timeout=300  # wait up to 5 minutes
)
# Schedule for later
from datetime import datetime, timedelta
flow_run = run_deployment(
    name="my-flow/my-deployment",
    scheduled_time=datetime.now() + timedelta(hours=2)
)
# With custom tags
flow_run = run_deployment(
    name="my-flow/my-deployment",
    tags=["production", "critical"]
)

```

By default, deployments triggered via `run_deployment` _from within another flow_ will be treated as a subflow of the parent flow in the UI. To disable this, set `as_subflow=False`.
### 
[​](https://docs.prefect.io/v3/how-to-guides/deployments/run-deployments#async-usage)
Async usage
In an async context, you can use the `run_deployment` function as a coroutine:
Copy
```
import asyncio
from prefect.deployments import run_deployment
async def trigger_deployment():
    flow_run = await run_deployment(
        name="my-flow/my-deployment",
        parameters={"my_param": 42}
    )
    return flow_run
# Run it
flow_run = asyncio.run(trigger_deployment())

```

## 
[​](https://docs.prefect.io/v3/how-to-guides/deployments/run-deployments#further-reading)
Further reading
  * Learn about [deployment schedules](https://docs.prefect.io/v3/how-to-guides/deployments/create-schedules)
  * Explore [deployment triggers and automations](https://docs.prefect.io/v3/how-to-guides/automations/creating-deployment-triggers)
  * Understand [work pools and workers](https://docs.prefect.io/v3/concepts/work-pools)


Was this page helpful?
YesNo
[Create Deployments](https://docs.prefect.io/v3/how-to-guides/deployments/create-deployments)[Create Deployment Schedules](https://docs.prefect.io/v3/how-to-guides/deployments/create-schedules)
Migrate
# How to Migrate from Airflow
Migration from Apache Airflow to Prefect: A Comprehensive How-To Guide
Migrating from Apache Airflow to Prefect simplifies orchestration, reduces overhead, and enables a more Pythonic workflow. Prefect’s flexible **library-based approach** lets you write, test, and run workflows with regular code—without the complexity of schedulers, executors, or metadata databases. This guide will walk you through a **step-by-step migration** , helping you transition from Airflow DAGs to Prefect flows while mapping key concepts, adapting infrastructure, and optimizing deployments. By the end, you’ll have a streamlined, scalable orchestration system that lets your team focus on engineering rather than maintaining workflow infrastructure. **Airflow to Prefect Mapping** This table provides a quick reference for migrating key Airflow concepts to their Prefect equivalents. Click on each concept to jump to a detailed explanation. **Airflow Concept** | **Prefect Equivalent** | **Key Differences**  
---|---|---  
[**DAGs**](https://docs.prefect.io/v3/how-to-guides/migrate/airflow#choose-a-dag-to-convert) | [**Flows**](https://docs.prefect.io/v3/how-to-guides/migrate/airflow#define-a-prefect-flow) | Prefect flows are standard Python functions (`@flow`). No DAG classes or `>>` dependencies.  
[**Operators**](https://docs.prefect.io/v3/how-to-guides/migrate/airflow#create-equivalent-prefect-tasks) | [**Tasks**](https://docs.prefect.io/v3/how-to-guides/migrate/airflow#create-equivalent-prefect-tasks) | Prefect tasks (`@task`) replace Airflow Operators, removing the need for specialized classes.  
[**Executors**](https://docs.prefect.io/v3/how-to-guides/migrate/airflow#airflow-executors) | [**Work Pools & Workers**](https://docs.prefect.io/v3/how-to-guides/migrate/airflow#airflow-executors) | Prefect decouples task execution using lightweight **workers** polling **work pools**.  
[**Scheduling**](https://docs.prefect.io/v3/how-to-guides/migrate/airflow#prefect-deployment) | [**Deployments**](https://docs.prefect.io/v3/how-to-guides/migrate/airflow#prefect-deployment) | Scheduling is separate from flow code and configured externally.  
[**XComs**](https://docs.prefect.io/v3/how-to-guides/migrate/airflow#define-a-prefect-flow) | [**Return Values**](https://docs.prefect.io/v3/how-to-guides/migrate/airflow#define-a-prefect-flow) | Prefect tasks return data directly; no need for XComs or metadata storage.  
[**Hooks & Connections**](https://docs.prefect.io/v3/how-to-guides/migrate/airflow#airflow-hooks-and-integrations) | [**Blocks & Integrations**](https://docs.prefect.io/v3/how-to-guides/migrate/airflow#airflow-hooks-and-integrations) | Prefect replaces Hooks with **Blocks** for secure resource management.  
[**Sensors**](https://docs.prefect.io/v3/how-to-guides/migrate/airflow#airflow-sensors) | [**Triggers & Event-Driven Flows**](https://docs.prefect.io/v3/how-to-guides/migrate/airflow#airflow-sensors) | Prefect uses external event triggers or lightweight polling flows.  
[**Airflow UI**](https://docs.prefect.io/v3/how-to-guides/migrate/airflow#observability) | [**Prefect UI**](https://docs.prefect.io/v3/how-to-guides/migrate/airflow#observability) | Prefect provides real-time monitoring, task logs, and automation features.  
There are also so key differences when it comes to task execution, resource control, data passing, and parallelism. We’ll cover these in more detail below. **Feature** | **Airflow** | **Prefect**  
---|---|---  
**Task Execution** | Tasks run as independent processes/pods | Tasks execute in single flow runtime  
**Resource Control** | Task-level via executor settings | Flow-level via work pools & task runners  
**Data Passing** | Requires XComs or external storage | Direct in-memory data passing  
**Parallelism** | Managed by executor configuration | Managed by work pools and task runners  
**Task Dependencies** | Uses `>>` operators and `set_upstream()` | Implicit via Python function calls  
**DAG Parsing** | Pre-parsed with global variable execution | Standard Python function execution  
**State & Retries** | Individual task retries, manual DAG fixes | Built-in flow & task retry handling  
**Scheduling** | Tightly coupled with DAG code | Decoupled via deployments  
**Infrastructure** | Requires scheduler, metadata DB, workers | Lightweight API server with optional cloud  
## 
[​](https://docs.prefect.io/v3/how-to-guides/migrate/airflow#preparing-for-migration)
Preparing for migration
Before jumping into code conversion, set the stage for a smooth migration. Preparation includes auditing your existing Airflow DAGs, setting up a Prefect environment for testing, and mapping Airflow concepts to their Prefect equivalents. **Audit your Airflow DAGs and dependencies:** Catalog all DAGs, schedules, task counts, and dependencies (databases, APIs, cloud services). Identify **high-priority pipelines** (business-critical, failure-prone, frequently updated) and **simpler DAGs** for pilot migration. Start with a small, non-critical DAG to gain confidence before tackling complex workflows. **Set up Prefect for testing:** Before fully migrating, set up a parallel Prefect environment to test your flows. Prefect provides a managed execution environment out of the box, so you can get started without configuring infrastructure.
  1. [**Install Prefect**](https://docs.prefect.io/v3/get-started/install) (`pip install prefect`).
  2. **Start a Prefect server locally** (`prefect server start`) or sign up for [**Prefect Cloud**](https://app.prefect.cloud/) to run flows immediately.
  3. **Run initial flows without infrastructure setup** : Run flows locally or using Prefect Cloud Managed Exxecution - allowing you to test without configuring work pools or Kubernetes.


Prefect Cloud provides a managed execution environment out of the box, so you can get started without configuring infrastructure.
Once you’ve validated basic functionality, you can explore configuring an [**execution environment**](https://docs.prefect.io/v3/deploy/infrastructure-concepts/work-pools) (e.g., Docker, Kubernetes) for production, which we cover later in this tutorial. For each Airflow DAG, you can outline its Prefect flow structure (tasks and control flow), where its schedule will live, and what execution infrastructure it needs. With preparation done, it’s time to start converting code.
## 
[​](https://docs.prefect.io/v3/how-to-guides/migrate/airflow#converting-dags-to-prefect-flows)
Converting DAGs to Prefect Flows
In this phase, you will **rewrite your Airflow DAGs as Prefect flows and tasks**. The goal is to replicate each workflow’s logic in Prefect, while simplifying wherever possible.
Prefect’s API is quite ergonomic - many Airflow users find they can express the same logic with _less code_ and _more flexibility_
Let’s break down the conversion process step-by-step, and walk through a concrete example.
### 
[​](https://docs.prefect.io/v3/how-to-guides/migrate/airflow#choose-a-dag-to-convert)
Choose a DAG to convert
Start with one of your simpler DAGs (perhaps one of those identified in the audit as an easy win). For illustration, suppose we have an Airflow DAG that runs a simple ETL: it **extracts data** , **transforms** it, and then **loads** the results. In Airflow, this might be defined as:
Copy
```
# Airflow DAG example (simplified ETL)
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
# Airflow task functions (to be used by PythonOperator)
def extract_fn():
    # ... (extract data, e.g., query an API or database)
    return data
def transform_fn(data):
    # ... (transform the data)
    return processed_data
def load_fn(processed_data):
    # ... (load data to target, e.g., save to DB or file)
with DAG("etl_pipeline", start_date=datetime(2023,1,1), schedule_interval="@daily", catchup=False) as dag:
    extract = PythonOperator(task_id='extract_data', python_callable=extract_fn)
    transform = PythonOperator(task_id='transform_data', python_callable=transform_fn)
    load = PythonOperator(task_id='load_data', python_callable=load_fn)
    # Set task dependencies
    extract >> transform >> load

```

In this Airflow DAG, we define three tasks using `PythonOperator`, then specify that they run sequentially (`extract` then `transform` then `load`).
### 
[​](https://docs.prefect.io/v3/how-to-guides/migrate/airflow#create-equivalent-prefect-tasks)
Create equivalent Prefect tasks
In Prefect, we’ll take the core logic of `extract_fn`, `transform_fn`, `load_fn` and turn each into a `@task` decorated function. The code inside can remain largely the same (minus any Airflow-specific cruft). For example:
Copy
```
# Prefect tasks for ETL
from prefect import task, flow
@task
def extract_data():
    # ... (extract data as before)
    return data
@task
def transform_data(data):
    # ... (transform data as before)
    return processed_data
@task
def load_data(processed_data):
    # ... (load data as before)

```

Notice we simply applied `@task` to each function. No need for a special operator class or task IDs - the function name serves as an identifier, and Prefect will handle the orchestration.
### 
[​](https://docs.prefect.io/v3/how-to-guides/migrate/airflow#define-a-prefect-flow)
Define a Prefect flow
Now we write a `@flow` function that calls these tasks in the required order:
Copy
```
@flow
def etl_pipeline():
    data = extract_data()         # calls extract_data task
    processed = transform_data(data)  # uses output of extract_data
    load_data(processed)          # calls load_data with result of transform_data

```

This Prefect flow function replaces the Airflow DAG. No need for `>>` dependencies or XComs. **Task results can be stored in variables that are passed directly to other tasks as arguments**. By default, tasks are automatically executed in the order they are called. Unlike Airflow, where testing often requires an Airflow context, Prefect flows run like standard Python code. You can execute `etl_pipeline()` in an interpreter, import it elsewhere, or test tasks individually (`transform_data.fn(sample_data)`).
## Key Differences
  * **Airflow:** Defines operators, sets dependencies (`>>`), and relies on XCom for data passing.
  * **Prefect:** Calls tasks like functions, with execution order determined by data flow, making workflows more intuitive and testable.


### 
[​](https://docs.prefect.io/v3/how-to-guides/migrate/airflow#branching-and-conditional-logic)
Branching and conditional logic
In Airflow, conditional branching is typically handled using BranchPythonOperator, ShortCircuitOperator, or trigger rules, requiring explicit DAG constructs to determine execution paths. Prefect simplifies branching by leveraging standard Python if/else logic directly within flows. **Implementing Branching in Prefect** Instead of using BranchPythonOperator and dummy tasks for joining paths, you can structure conditional execution using native Python control flow:
Copy
```
@flow
def my_flow():
    result = extract_data()
    if some_condition(result):
        outcome = branch_task_a()  # a task or subflow for branch A
    else:
        outcome = branch_task_b()  # branch B
    final_task(outcome)

```

**Key Differences from Airflow** **Feature** | **Airflow (BranchPythonOperator)** | **Prefect (`if/else` logic)**  
---|---|---  
**Branching Method** | Often uses specialized operators (`BranchPythonOperator`) | Uses native Python conditionals (`if/else`)  
**Skipped Tasks** | Unselected branches are explicitly **skipped** | Prefect **only runs** the executed branch—no skipping needed  
**Join Behavior** | Uses **DummyOperator** to rejoin paths | Downstream tasks execute **automatically** after the conditional branch  
**Advantages of Prefect’s Approach**
  * **No special operators** — branching is simpler and more intuitive
  * **Cleaner code** — fewer unnecessary tasks like `DummyOperator`
  * **No explicit skipping required** — Prefect only executes the called tasks

By using standard Python control flow, Prefect **eliminates complexity** and makes conditional execution more **readable, maintainable, and testable**.
### 
[​](https://docs.prefect.io/v3/how-to-guides/migrate/airflow#retries-and-error-handling)
Retries and error handling
Airflow DAGs often have retry settings either at the DAG level (`default_args`) or per task (e.g., `retries=3`). In Prefect, you can specify [retries](https://docs.prefect.io/v3/develop/write-flows#retries) for any task or flow. Use `@task(retries=2, retry_delay_seconds=60)` to retry a task twice on failure, or `@flow(retries=1)` to retry the entire flow once. Prefect **distinguishes flow and task retries** —flow retries rerun all tasks, while task retries rerun only the failed task. Replace Airflow-specific error handling (`on_failure_callback`, sensors) with Prefect’s **Retry** , **State Handlers** , or built-in failure notifications.
### 
[​](https://docs.prefect.io/v3/how-to-guides/migrate/airflow#remove-airflow-specific-code)
Remove Airflow-specific code
Go through the DAG code and strip out anything that doesn’t apply in Prefect. This includes: DAG declarations (`DAG(...)` blocks), default_args, Airflow imports (`from airflow...`), XCom push/pull calls (replace with return values), Jinja templating in operator arguments (you can often just compute those values in Python directly or use [Prefect parameters](https://docs.prefect.io/v3/deploy/index#workflow-scheduling-and-parametrization)). If your DAG used Airflow Variables or Connections (Airflow’s way to store config in the Metastore), you’ll need to supply those to Prefect tasks via another means - for example, as [environment variables](https://docs.prefect.io/v3/develop/settings-and-profiles#environment-variables) or using [Prefect Blocks](https://docs.prefect.io/integrations/integrations) (like a Block for a database connection string). Essentially, your Prefect flow code should look like a regular Python script with functions, not like an Airflow DAG file. As an illustration, here’s how our example **ETL pipeline** looks after conversion:
Copy
```
from prefect import flow, task
@task(retries=1, log_prints=True)
def extract_data():
    # fetch data from API (simulated)
    data = get_data_from_api()
    return data
@task
def transform_data(data):
    # process the data
    processed = transform(data)
    return processed
@task
def load_data(data):
    # load data to database
    load_into_db(data)
@flow(name="etl_pipeline")
def etl_pipeline_flow():
    raw = extract_data()
    processed = transform_data(raw)
    load_data(processed)
if __name__ == "__main__":
    # For local testing
    etl_pipeline_flow()

```

Key improvements in this converted code:
  * **Direct execution for testing** - `if __name__ == "__main__": etl_pipeline_flow()` allows running the flow locally during development. In production, Prefect handles scheduling.
  * **Built-in retries and logging** - `retries=1` ensures one retry on failure, and `log_prints=True` sends `print()` output to Prefect’s UI.
  * **Pure Python** - No Airflow imports or context, making the flow easy to test, debug, and run consistently across environments (IDE, CI, or production).


### 
[​](https://docs.prefect.io/v3/how-to-guides/migrate/airflow#validate-functional-equivalence)
Validate functional equivalence
Once a DAG has been rewritten as a Prefect flow, execute the flow and compare its results with the Airflow DAG to ensure expected outcomes. If discrepancies arise, modify the flow accordingly. Keep in mind the original DAG may have depended on XComs or global variables that you will need to account for. For each task and special case, including [subDAGs](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/dags.html#concepts-subdags) and [TaskGroups](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/dags.html#taskgroups), implement them as subflows or Python functions in Prefect. When transitioning from Airflow’s TaskFlow API, keep in mind that Prefect’s `@task` decorator serves a similar purpose but does not rely on XComs. After completing these steps, the Prefect flow should accurately replicate the functionality of the Airflow DAG while being more modular and testable. The migration is now complete, and the next step is to focus on deploying and optimizing the new workflows.
## 
[​](https://docs.prefect.io/v3/how-to-guides/migrate/airflow#infrastructure-migration-considerations)
Infrastructure Migration Considerations
Migrating your code is a big step, but ensuring your workflows run smoothly in Prefect is just as important. Prefect’s **flexible execution** makes this easier, supporting Prefect managed execution, local machines, VMs, containers, and Kubernetes with less setup. This section maps Airflowss executors to **Prefect Work Pools and Workers** , while also covering sensors, hooks, logging, and state management to complete your migration.
### 
[​](https://docs.prefect.io/v3/how-to-guides/migrate/airflow#leveraging-prefect-managed-execution)
Leveraging Prefect Managed Execution
**Running Flows Without Infrastructure Setup** Prefect Cloud offers [Managed Execution](https://docs.prefect.io/v3/how-to-guides/deployment_infra/serverless), allowing you to run flows **without setting up infrastructure or maintaining workers**. With Prefect Managed work pools, Prefect handles compute, execution, and scheduling, eliminating the need for a cloud provider account or on-premises infrastructure. **Getting Started with Prefect Managed Execution**
1
Create a Prefect Managed Work Pool
Copy
```
prefect work-pool create my-managed-pool --type prefect:managed

```

2
Deploy a Flow to Managed Execution
Copy
```
from prefect import flow
if __name__ == "__main__":
    flow.from_source(
        source="https://github.com/prefecthq/demo.git",
        entrypoint="flow.py:my_flow",
    ).deploy(
        name="test-managed-flow",
        work_pool_name="my-managed-pool",
    )

```

3
Run the Deployment via Prefect UI or CLI
Copy
```
python managed-execution.py

```

This will allow your flow to run remotely without provisioning workers, setting up Kubernetes, or maintaining cloud infrastructure. **When to Use Prefect Managed Execution**
## Best for
Ideal for testing and running flows without infrastructure setup, especially for teams that want managed execution without a cloud provider.
## Consider self-hosted execution
If you need custom images, heavy dependencies, private networking, or higher concurrency limits than Prefect’s tiers allow.
**Next Steps** If you require self-hosted execution, the next sections cover how to migrate Airflow Executors to Prefect Work Pools across different infrastructure types (Kubernetes, Docker, Celery, etc.). For full details on Prefect Managed Execution, refer to the [Managed Execution documentation](https://docs.prefect.io/v3/how-to-guides/deployment_infra/serverless).
### 
[​](https://docs.prefect.io/v3/how-to-guides/migrate/airflow#airflow-executors)
Airflow Executors
**Airflow Executors vs Prefect Work Pools/Workers:** Airflow’s executor setting determines how tasks are distributed. Prefect’s equivalent concept is the [**work pool**](https://docs.prefect.io/v3/deploy/infrastructure-concepts/work-pools) (with one or more [**workers**](https://docs.prefect.io/v3/deploy/infrastructure-concepts/workers) polling it). In Airflow, each task executes independently, regardless of the executor used. Whether running with LocalExecutor, CeleryExecutor, or KubernetesExecutor, every task runs as an isolated process or pod. Executors control how and where these tasks are executed, but the core execution model remains task-by-task. In contrast, Prefect executes an entire flow run within a single execution environment (e.g., a local process, Docker container, or Kubernetes pod). Tasks within a flow execute within the same runtime context, reducing fragmentation and improving performance. Prefect’s execution model simplifies resource management, allowing for in-memory data passing between tasks rather than relying on external storage or metadata databases. Here’s a mapping of typical setups:
#### 
[​](https://docs.prefect.io/v3/how-to-guides/migrate/airflow#airflow-localexecutor)
Airflow LocalExecutor
With the **Airflow LocalExecutor** tasks run as subprocesses on the same machine. In Prefect, the default behavior is similar - you can run the flow in a local Python process, and tasks will execute sequentially by default. That does not _have_ to be the same machine that is running your Prefect UI and scheduler. For parallelism on a single machine, use [**`DaskTaskRunner`**](https://docs.prefect.io/integrations/prefect-dask/index)to enable multi-process execution:
Copy
```
@flow(task_runner=DaskTaskRunner())

```

By default, Prefect’s **Process work pool** runs flows as subprocesses. A basic **Airflow LocalExecutor** setup can be replaced with a **Prefect worker** on the same VM using a **process work pool** , eliminating the need for a separate scheduler.
#### 
[​](https://docs.prefect.io/v3/how-to-guides/migrate/airflow#airflow-celeryexecutor)
Airflow CeleryExecutor
**Airflow CeleryExecutor** where distributed workers run across multiple machines, using a message broker like RabbitMQ/Redis. Prefect eliminates the need for a **message broker** or **results backend** , as its API server manages work distribution. To replicate an Airflow **CeleryExecutor** setup, deploy **multiple Prefect workers** across machines, all polling from a shared **work pool**. **Setting Up a Work Pool and Workers**
  1. **Create a work pool** (e.g., `"prod-work-pool"`): 
Copy
```
prefect work-pool create prod-work-pool --type process

```

  2. **Start a worker on each node** , assigning it to the work pool: 
Copy
```
prefect worker start -p prod-work-pool

```

  3. **Workers poll the work pool** and execute assigned flow runs.

Prefect **work pools** function similarly to **Celery queues** , allowing multiple workers to process tasks concurrently.
#### 
[​](https://docs.prefect.io/v3/how-to-guides/migrate/airflow#airflow-kubernetesexecutor)
Airflow KubernetesExecutor
In Airflow, the **KubernetesExecutor** follows the per-task execution model, launching each task in its own Kubernetes pod. Prefect, instead, uses a Kubernetes Work Pool, where each flow run executes in a single Kubernetes pod. This approach reduces fragmentation, as tasks run within the same execution environment rather than spawning separate pods. **Configuring a Kubernetes Work Pool** For detailed instructions, see [Prefect’s Kubernetes Work Pool documentation](https://docs.prefect.io/v3/how-to-guides/deployment_infra/kubernetes). But the general steps to take are:
  1. **Create a Kubernetes work pool** with the desired pod template (e.g., image, resources): 
Copy
```
prefect work-pool create k8s-pool --type kubernetes

```

  2. **Deploy a flow to the Kubernetes work pool** :


Copy
```
from prefect import flow
@flow(log_prints=True)
def buy():
    print("Buying securities")
if __name__ == "__main__":
    buy.deploy(
        name="my-code-baked-into-an-image-deployment",
        work_pool_name="k8s-pool",
        image="my_registry/my_image:my_image_tag"
    )

```

Alternatively, you can use a [prefect.yaml](https://docs.prefect.io/v3/how-to-guides/deployment_infra/kubernetes#define-a-prefect-deployment) file to deploy your flow to the Kubernetes work pool.
  1. [**Run a Kubernetes worker in-cluster**](https://docs.prefect.io/v3/how-to-guides/deployment_infra/kubernetes#deploy-a-worker-using-helm) to execute flow runs.
  2. **Execution Flow** :


  * The worker **picks up a scheduled flow run**.
  * It **creates a new pod** , which executes the entire flow.
  * The **pod terminates automatically** after execution.

This setup eliminates the need for a long-running scheduler, reducing operational complexity while leveraging Kubernetes for **on-demand, containerized execution**.
#### 
[​](https://docs.prefect.io/v3/how-to-guides/migrate/airflow#airflow-celerykubernetes)
Airflow CeleryKubernetes
**Airflow + Celery + Kubernetes (CeleryKubernetes Executor)** or other hybrid: Some Airflow deployments use Celery for distributed scheduling but run tasks in containers or on Kubernetes. Prefect’s model can handle these as well by combining approaches - e.g., use a Kubernetes work pool with multiple worker processes distributed as needed. The general principle is that Prefect **work pools** can cover all these patterns (local, multi-machine, containers, serverless) via configuration, not code, and you manage them via Prefect’s UI/CLI.
#### 
[​](https://docs.prefect.io/v3/how-to-guides/migrate/airflow#using-serverless-compute)
Using Serverless compute
Prefect supports [**serverless execution**](https://docs.prefect.io/v3/how-to-guides/deployment_infra/serverless) on various cloud platforms, eliminating the need for dedicated infrastructure. Instead of provisioning long-running workers, flows can be executed **on-demand** in ephemeral environments. Prefect’s push-based work pools allow flows to be submitted to serverless services, where they run in isolated containers and automatically scale with demand. **Serverless Platforms** Prefect flows can run on:
  * **AWS ECS** (Fargate or EC2-backed containers)
  * **Azure Container Instances (ACI)**
  * **Google Cloud Run**
  * **Modal** (serverless compute for AI/ML workloads)
  * **Coiled** (serverless Dask clusters for parallel workloads)

**Configuring a Serverless Work Pool** To run flows on a serverless platform, create a **push-based work pool** and configure it to submit jobs to the desired service. Example: Creating an **ECS work pool** :
Copy
```
prefect work-pool create --type ecs:push --provision-infra my-ecs-pool

```

Deployments can then be configured to use the serverless work pool, allowing Prefect to submit flow runs without maintaining long-lived infrastructure. For setup details, refer to [Prefect’s serverless execution documentation](https://docs.prefect.io/v3/how-to-guides/deployment_infra/serverless).
### 
[​](https://docs.prefect.io/v3/how-to-guides/migrate/airflow#airflow-sensors)
Airflow Sensors
Airflow Sensors continuously poll for external conditions, such as file availability or database changes, which can tie up resources. Prefect replaces this with an **event-driven approach** , where external systems trigger flow execution when conditions are met. **Using External Triggers** Instead of using an Airflow `S3KeySensor`, configure an AWS Lambda or EventBridge rule to call the Prefect API when an S3 file is uploaded. Prefect Cloud and Server provide API endpoints to start flows on demand. Prefect’s **Automations** can also trigger flows based on specific conditions. **Handling Polling Scenarios** If an external system lacks event-driven capabilities, implement a lightweight **polling flow** that runs on a schedule (e.g., every 5 minutes), checks the condition, and triggers the main flow if met. This approach minimizes idle resource consumption compared to Airflow’s persistent sensors. Prefect’s model eliminates long-running sensor tasks, making workflows **more efficient, scalable, and event-driven**.
### 
[​](https://docs.prefect.io/v3/how-to-guides/migrate/airflow#airflow-hooks-and-integrations)
Airflow Hooks and Integrations
Airflow provides hooks and operators for interacting with external systems (e.g., **JDBC, cloud services, databases**). In Prefect, these integrations are handled through [**Prefect Integrations**](https://docs.prefect.io/integrations/integrations) (e.g., `prefect-snowflake`, `prefect-gcp`, `prefect-dbt`) or by directly using the relevant **Python libraries** within tasks. **Migrating Airflow Hooks to Prefect**
  1. **Identify Airflow hooks** used in your DAGs (e.g., `PostgresHook`, `GoogleCloudStorageHook`).
  2. **Replace them with equivalent Prefect integrations** or direct Python library calls.

**Example:** Instead of
Copy
```
hook = PostgresHook(postgres_conn_id=my_conn_id)
engine = hook.get_sqlalchemy_engine()
session = sessionmaker(bind=engine)()

```

Use Prefect Blocks for secure credential management:
Copy
```
from prefect_sqlalchemy import SqlAlchemyConnector
SqlAlchemyConnector.load("BLOCK_NAME-PLACEHOLDER")

```

  1. **Use Prefect Blocks for secrets management** , similar to Airflow Connections, to separate credentials from code.

**Replacing Airflow Operators with Prefect Tasks**
  * **Prefect tasks** can call any Python library, eliminating the need for custom Airflow operators.
  * Example: Instead of using a **BashOperator** to call an API via a shell script, install the necessary package in the flow’s environment and call it directly in a task.

Prefect’s approach **removes unnecessary abstraction layers** , allowing direct access to the full Python ecosystem without Airflow-specific constraints. Basically: **anything done with a custom Airflow operator or hook can be replaced in Prefect with a task using the appropriate Python library.** Prefect removes Airflow’s constraints, allowing direct use of the full Python ecosystem. For example, instead of using a **BashOperator** to call an API via a shell script, install the required package in your environment and call it directly from a task, eliminating unnecessary workarounds.
### 
[​](https://docs.prefect.io/v3/how-to-guides/migrate/airflow#observability)
Observability
#### 
[​](https://docs.prefect.io/v3/how-to-guides/migrate/airflow#state-and-logging)
State and logging
**Task and Flow State Management** In Airflow, task states (`success`, `failed`, `skipped`, etc.) are stored in a metadata database and displayed in the Airflow UI’s DAG run view. Prefect also tracks state for **each task and flow run** , but these states are managed by the **Prefect backend** (Prefect Server or Cloud API) and can be accessed via the **Prefect UI, API, or CLI**. After migration, similar visibility is available in Prefect’s UI, where you can track which flows and tasks succeeded or failed. Prefect also includes additional state management features such as:
  * Cancel a flow run (`Cancelling` state).
  * Retry a failed flow run (with manual steps).
  * **Task caching** between runs to avoid redundant computations.

**Logging Differences** Airflow logs task execution output to files (stored on executor machines or remote storage), viewable through the UI. Prefect **captures stdout, stderr, and Python logging** from tasks and sends them to the Prefect backend, making logs accessible in the **Prefect UI, API, and CLI**. To ensure logs appear correctly in Prefect’s UI, use `@flow(log_prints=True)` or `@task(log_prints=True)` These flags route `print()` statements to Prefect logs automatically. For centralized logging (e.g., ElasticSearch, Stackdriver), Prefect supports [**custom logging handlers**](https://docs.prefect.io/v3/advanced/logging-customization) and **third-party integrations**. Logs can be forwarded similarly to how Airflow handled external logging. **Debugging and Troubleshooting** Prefect simplifies debugging because tasks are **standard Python functions**. Instead of analyzing scheduler or worker logs, you can:
  * **Re-run individual tasks or flows locally** to reproduce issues.
  * **Test flows interactively** in an IDE before deploying.

This direct execution model eliminates the need to troubleshoot failures through a scheduling system, making debugging faster and more intuitive than in Airflow.
#### 
[​](https://docs.prefect.io/v3/how-to-guides/migrate/airflow#monitoring)
Monitoring
**Notifications and Alerts** In Airflow, monitoring is typically managed through the UI, email alerts on task failures, and external monitoring of the scheduler. Prefect provides similar capabilities through _Automations_ , which can be configured to trigger alerts via Slack, email, or webhooks based on specific events. To replicate Airflow’s alerting (e.g., failures or SLA misses), configure [**Prefect Automations**](https://docs.prefect.io/v3/automate/events/automations-triggers) to:
  * Notify on **flow or task failures**.
  * Alert when a **flow run exceeds a specified runtime**.
  * Trigger **custom actions** based on state changes.

**Service Level Agreements (SLAs)** Prefect Cloud supports Service Level Agreements (SLAs) to monitor and enforce performance expectations for flow runs. SLAs automatically trigger alerts when predefined thresholds are violated. SLAs can be defined via the Prefect UI, prefect.yaml, `.deploy()` method, or CLI. Violations generate `prefect.sla.violation` events, which can trigger Automations to send notifications or take corrective actions. For full configuration details, refer to the [Measure reliability with Service Level Agreements](https://docs.prefect.io/v3/automate/events/slas) documentation. **Implementation Considerations** Prefect allows flexible logging and alerting adjustments to match existing monitoring workflows. Logging handlers can integrate with **third-party services** (e.g., ElasticSearch, Datadog), and **Prefect’s API and UI provide real-time state visibility** for proactive monitoring.
## 
[​](https://docs.prefect.io/v3/how-to-guides/migrate/airflow#deployment-%26-ci%2Fcd-changes)
Deployment & CI/CD Changes
Deploying workflows in Prefect differs from Airflow’s approach of “drop DAG files in a folder.” In Prefect, a **Deployment** is the unit of deployment: it associates a flow (Python function) with infrastructure (how/where to run) and optional schedule or triggers. Migrating to Prefect means adopting a new way to package and release your workflows, as well as updating any CI/CD pipelines that automated your Airflow deployments.
### 
[​](https://docs.prefect.io/v3/how-to-guides/migrate/airflow#prefect-deployment)
Prefect Deployment
**From Airflow DAG schedules to Prefect Deployment:** In Airflow, deployment usually meant placing your DAG code on the Airflow scheduler (e.g., by committing to a Git repo that the scheduler reads, or copying files to the DAGs directory). There isn’t a formal deployment artifact beyond the Python files. Prefect, by contrast, treats deployments as first-class objects. You will create a deployment for each flow (or for each distinct configuration of a flow you want to run). This can be done via code (calling `flow.deploy()`), via CLI (`prefect deployment`), or by writing a YAML (`prefect.yaml`) that describes the deployment. Key things a **Prefect deployment** defines:
  * **Target flow** (which function, and which file or import path it comes from).
  * **Infrastructure configuration** : e.g., use the “Kubernetes work pool” or “process” type, possibly the docker image to use, resource settings, etc.
  * **Storage of code** : e.g., whether the code is stored in the image, pulled from Git, etc. (Prefect can package code into a Docker image or rely on an existing image).
  * **Schedule** (optional): e.g., Cron or interval schedule for automatic runs, or you can leave it manual.
  * **Parameters** (optional): default parameter values for the flow, if any.

To migrate each Airflow DAG, you will create a Prefect deployment for its flow. For example, if we converted `etl_pipeline` DAG to `etl_pipeline_flow` in Prefect, we might write a `prefect.yaml` like:
Copy
```
# prefect.yaml
deployments:
  - name: etl-pipeline-prod
    flow_name: etl_pipeline_flow
    entrypoint: etl_flow.py:etl_pipeline_flow # file and function where the flow is defined
    parameters: {}
    schedule: "@daily"
    work_pool:
      name: prod-k8s-pool
      # other infra settings like image, etc., if needed

```

This YAML can define multiple deployments, but in this case we have one named “etl-pipeline-prod” which runs daily via the `prod-k8s-pool` (a Kubernetes pool perhaps). In Airflow, these details were all intertwined in the DAG file (the schedule was in code, the infrastructure maybe in the executor config or the DAG via `executor_config`). In Prefect, there is a separation of these concerns.
### 
[​](https://docs.prefect.io/v3/how-to-guides/migrate/airflow#automation-via-ci%2Fcd)
Automation via CI/CD
Many organizations use CI/CD to deploy Airflow DAGs (for example, a Git push triggers a Jenkins job that lints DAGs and copies them to the Airflow server). With Prefect, you’ll likely adjust your CI/CD to **register Prefect deployments** whenever you update the flow code. Prefect’s CLI is your friend here. A common pattern is:
  1. On merge to main, build a Docker image with your flow code, push it to a registry
  2. Then run `prefect deployment build -n <name> -p <work_pool_name> --cron "<schedule>" -q default -o deployment.yaml` (or use `prefect.yaml`) and apply it.

This can all be scripted. In fact, Prefect provides guidance on using [GitHub Actions or similar tooling to do this](https://docs.prefect.io/v3/advanced/deploy-ci-cd). By integrating Prefect’s deployment steps into CI, you ensure that any change in your flow code gets reflected in Prefect’s orchestrator, much like updating DAG code in Airflow. Alternatively, if your deployment is set to pull the workflow code from your git repository each time, you only need to push the latest workflow code, and automatically next time your deployment runs it will pull the latest workflow code. This CI pipeline approach allows versioning and automating your flows deployment, treating them similarly to application code deployments. It’s a shift from Airflow where deployment could be syncing a folder - Prefect’s method is more **controlled** and **atomic** (you create a deployment manifest and apply it, which registers everything with Prefect).
### 
[​](https://docs.prefect.io/v3/how-to-guides/migrate/airflow#prefect-in-production)
Prefect in Production
Once deployed, Prefect schedules and orchestrates flows based on your **deployments**. Follow these best practices to ensure a reliable production setup:
  * **High Availability** : If self-hosting, use PostgreSQL and consider running **multiple API replicas** behind a load balancer. [**Prefect Cloud**](https://prefect.io/cloud) handles availability automatically.
  * **Keep Workers Active** : Ensure Prefect workers are always running, whether as systemd services, Docker containers, or Kubernetes deployments.
  * **Logging & Observability**: Use Prefect’s UI for logs or configure external storage (e.g., S3, Elasticsearch) for **long-term retention**.
  * **Notifications & Alerts**: Set up failure alerts via Slack, email, or Twilio using [**Prefect Automations**](https://docs.prefect.io/v3/automate/events/automations-triggers) to ensure timely issue resolution.
  * **CI/CD & Testing**: Validate deployment YAMLs in CI (`prefect deployment build --skip-upload`), and unit test tasks as regular Python functions.
  * **Configuration Management** : Replace Airflow Variables/Connections with [**Prefect Blocks**](https://docs.prefect.io/v3/develop/variables), storing secrets via CLI, UI, or version-controlled JSON.
  * **Security & Access Control**: Prefect Cloud includes built-in authentication & role-based access; self-hosted setups should secure API and workers accordingly.
  * **Decommissioning Airflow** : Once migration is complete, disable DAGs, archive the code, and shut down Airflow components to reduce operational overhead.

For more details on operating Prefect in production, see the [How-To Guides](https://docs.prefect.io/v3/how-to-guides).
## 
[​](https://docs.prefect.io/v3/how-to-guides/migrate/airflow#testing-%26-validation)
Testing & Validation
Thorough testing ensures your Prefect flows perform like their Airflow equivalents. Since this is a **full migration** , validation is essential before decommissioning Airflow. **Testing Prefect Flows in Isolation**
  * **Unit test task logic** - Write tests for tasks as regular Python functions.
  * **Run flows locally** - Run the script that calls your flow function - just like a normal Python script.
  * **Use Prefect’s local orchestration** - Start a Prefect server (`prefect server start`), register a deployment, and trigger flows via Prefect UI to mirror production behavior.
  * **Compare outputs** - Run both Airflow and Prefect for the same input and validate results (e.g., database rows, file outputs). Debug discrepancies early.

**Validation Phase: Temporary Parallel Running (Shadow Mode)**
  * **Keep the Airflow DAG inactive** but available for testing.
  * **Manually trigger** both Airflow and Prefect flows for the same execution date.
  * **Write test outputs separately** to prevent conflicts, ensuring parity before stopping Airflow runs.

For batch jobs, this phase should be **brief** , ensuring correctness without long-term dual maintenance. **Decommissioning Airflow** Once a Prefect flow is stable, **disable the corresponding Airflow DAG** to prevent accidental execution. Clearly document Prefect as the new source of truth. Avoid keeping inactive DAGs indefinitely, as they can cause confusion—**archive or remove them once the migration is complete**.
### 
[​](https://docs.prefect.io/v3/how-to-guides/migrate/airflow#common-issues-and-troubleshooting)
Common issues and troubleshooting
  * **Missing dependencies:** If a Prefect flow fails with `ImportError`, ensure all required libraries are installed in the execution environment (Docker image, VM, etc.), not just locally.
  * **Credentials & access:** Verify that Prefect workers have the same permissions as Airflow (e.g., service accounts, IAM roles). If using Kubernetes, ensure pods can access necessary databases and APIs.
  * **Scheduling differences:** Airflow schedules may trigger at the end of an interval, while Prefect runs in real-time. Align Cron schedules and time zones if needed.
  * **Concurrency & parallelism:** Configure **work pool and flow run concurrency limits** to prevent overlapping jobs. If too many tasks run in parallel, use Prefect’s **tags and concurrency controls** to throttle execution.
  * **Error handling & retries:** Test retries by forcing failures. If Airflow used `trigger_rule="all_done"`, implement equivalent logic in Prefect with `try/except`.
  * **Performance monitoring:** Compare Prefect vs. Airflow run times. If slower, check if tasks are running sequentially instead of in parallel (enable mapping, async, or parallel task runners). If too much parallelism, adjust concurrency settings.

For some help with troubleshooting, you can see articles on:
  * [Configuring logging](https://docs.prefect.io/v3/how-to-guides/workflows/add-logging)
  * [Tracking activity](https://docs.prefect.io/v3/concepts/events)

Throughout testing, keep an eye on the Prefect UI’s **Flow Run and Task Run views** - they will show you the execution steps, logs, and any exceptions. The UI can be very helpful for pinpointing where a flow failed or hung. It’s analogous to Airflow’s Graph view and log view but with the benefit of real-time state updates (no need to refresh for state changes). You might also consider joining the [Prefect Slack community](https://prefect.io/slack) to get help from the community and Prefect team. **Debugging tips:**
  * If a flow run gets stuck, you can cancel it via UI/CLI.
  * Utilize the fact that you can re-run a Prefect flow easily. For example, if a specific task fails consistently, you can add some debug `print` statements, re-deploy (which is quick with Prefect CLI), and re-run to see output.
  * Leverage Prefect’s task state inspection. In the UI, you can often see the exception message and stack trace for a failed task, which helps identify the problem in code.
  * Read the results from MarvinAI’s analysis of your code to help identify potential issues.


MarvinAI is a tool that can help you debug your Prefect flows.
As you systematically validate each migrated workflow, you’ll build confidence in the new system. When all tests pass and the outputs match the old system’s, you can declare the migration a success for that workflow. After migrating a few, you’ll also develop a playbook for the rest, and the process may speed up.
## 
[​](https://docs.prefect.io/v3/how-to-guides/migrate/airflow#post-migration)
Post-Migration
### 
[​](https://docs.prefect.io/v3/how-to-guides/migrate/airflow#optimizing-%26-scaling-prefect-workflows)
Optimizing & Scaling Prefect Workflows
With your workflows running in Prefect, it’s time to optimize, scale, and take full advantage of its capabilities. This section covers best practices for streamlining flows, monitoring performance, and ensuring long-term reliability. **Simplify and Enhance Your Workflows**
  * **Remove unnecessary complexity** : If your Airflow DAGs used workarounds (e.g., database intermediaries for data passing), replace them with direct Prefect task returns.
  * **Use nested flows for modularity** : Instead of chaining DAGs, use **nested flows** to orchestrate dependencies within a single flow.
  * **Optimize async convenience** : Use **dynamic task mapping** (`task.map(items)`) to process large datasets efficiently.
  * **Leverage caching** : Enable **result persistence** to skip redundant computations.
  * **Ensure idempotency** : Prevent duplicate processing by parameterizing flows and validating execution logic.

**Monitor and Maintain Your Prefect System**
  * **Track performance** : Use Prefect UI and analytics to monitor run durations, failure rates, and bottlenecks.
  * **Set up alerts** : Automate failure notifications via Slack, email, or other integrations.
  * **Improve debugging** : Use UI logs, parameterized re-runs, and version control for better issue resolution.
  * **Version control deployments** : Treat flows like code, using PRs and staging environments before production deployment.
  * **Update documentation** : Ensure internal runbooks reflect Prefect’s CLI/UI for managing schedules, failures, and retries.

To scale and optimize for cost: Technique | Description  
---|---  
_Scale efficiently_ | Prefect makes it simple to distribute workloads across work pools and workers, eliminating Airflow’s scheduler bottlenecks.  
_Optimize infrastructure_ | Adjust worker capacity based on usage, scaling vertically (more resources per worker) or horizontally (adding more workers).  
_Reduce costs_ | Consider _serverless work pools_ (AWS ECS, GCP Cloud Run) to avoid idle infrastructure costs.  
To set yourself up for future success: Technique | Description  
---|---  
_Share best practices_ | Conduct a team retrospective to refine workflows and establish templates for new flows.  
_Embrace Prefect’s flexibility_ | Now that scheduling and execution are handled seamlessly, focus on building better data workflows, not managing infrastructure.  
### 
[​](https://docs.prefect.io/v3/how-to-guides/migrate/airflow#conclusion)
Conclusion
By completing this migration, you’ve moved to a more scalable, efficient orchestration system. Prefect allows your team to focus on engineering—iterating faster, improving reliability, and scaling seamlessly. **Next steps:**
  * [Learn more about Prefect](https://docs.prefect.io/v3/get-started/index)
  * [See more Prefect examples](https://docs.prefect.io/v3/examples/index)
  * [Join the community](https://prefect.io/slack)
  * [Dig into Prefect’s Integrations](https://docs.prefect.io/integrations/integrations)
  * [Learn more about Prefect Cloud](https://prefect.io/cloud)
  * [Visit Prefect’s GitHub](https://github.com/PrefectHQ/prefect)


Was this page helpful?
YesNo
[Run the Prefect Server via Docker Compose](https://docs.prefect.io/v3/how-to-guides/self-hosted/docker-compose)[Upgrade to Prefect 3.0](https://docs.prefect.io/v3/how-to-guides/migrate/upgrade-to-prefect-3)
Migrate
# How to upgrade from agents to workers
Learn how to upgrade from agents to workers to significantly enhance the experience of deploying flows.
Upgrading from agents to workers significantly enhances the experience of deploying flows by simplifying the specification of each flow’s infrastructure and runtime environment.
This guide is for users who are upgrading from agents (a deployment pattern specific to `prefect>2.0,<3.0`) to workers. If you are new to Prefect, we recommend starting with the [Prefect Quickstart](https://docs.prefect.io/v3/get-started/quickstart).
## 
[​](https://docs.prefect.io/v3/how-to-guides/migrate/upgrade-agents-to-workers#about-workers-and-agents)
About workers and agents
A [worker](https://docs.prefect.io/v3/concepts/workers) is the fusion of an agent with an infrastructure block. Like agents, workers poll a work pool for flow runs that are scheduled to start. Like infrastructure blocks, workers are typed. They work with only one kind of infrastructure, and they specify the default configuration for jobs submitted to that infrastructure. Accordingly, workers are not a drop-in replacement for agents. **Using workers requires deploying flows differently.** In particular, deploying a flow with a worker does not involve specifying an infrastructure block. Instead, infrastructure configuration is specified on the [work pool](https://docs.prefect.io/v3/concepts/work-pools) and passed to each worker that polls work from that pool.
## 
[​](https://docs.prefect.io/v3/how-to-guides/migrate/upgrade-agents-to-workers#upgrade-enhancements)
Upgrade enhancements
### 
[​](https://docs.prefect.io/v3/how-to-guides/migrate/upgrade-agents-to-workers#workers)
Workers
  * Improved visibility into the status of each worker, including when a worker was started and when it last polled.
  * Better handling of race conditions for high availability use cases.


### 
[​](https://docs.prefect.io/v3/how-to-guides/migrate/upgrade-agents-to-workers#work-pools)
Work pools
  * Work pools allow greater customization and governance of infrastructure parameters for deployments through their [base job template](https://docs.prefect.io/v3/how-to-guides/deployment_infra/manage-work-pools#base-job-template).
  * Prefect Cloud [push work pools](https://docs.prefect.io/v3/how-to-guides/deployment_infra/serverless) enable flow execution in your cloud provider environment without the need to host a worker.
  * Prefect Cloud [managed work pools](https://docs.prefect.io/v3/how-to-guides/deployment_infra/managed) allow you to run flows on Prefect’s infrastructure, without the need to host a worker or configure cloud provider infrastructure.


### 
[​](https://docs.prefect.io/v3/how-to-guides/migrate/upgrade-agents-to-workers#improved-deployment-interfaces)
Improved deployment interfaces
  * The Python deployment experience with [`.deploy()`](https://docs.prefect.io/v3/how-to-guides/deployments/deploy-via-python) or the alternative deployment experience with `prefect.yaml` are more flexible and easier to use than block and agent-based deployments.
  * Both options allow you to [deploy multiple flows](https://docs.prefect.io/v3/deploy/infrastructure-concepts/prefect-yaml#work-with-multiple-deployments-with-prefect-yaml) with a single command.
  * Both options allow you to build Docker images for your flows to create portable execution environments.
  * The YAML-based API supports [templating](https://docs.prefect.io/v3/deploy/infrastructure-concepts/prefect-yaml#templating-options) to enable [dryer deployment definitions](https://docs.prefect.io/v3/how-to-guides/deployments/prefect-yaml#reuse-configuration-across-deployments).


## 
[​](https://docs.prefect.io/v3/how-to-guides/migrate/upgrade-agents-to-workers#upgrade-changes)
Upgrade changes
  1. **Deployment CLI and Python SDK:** `prefect deployment build <entrypoint>`/`prefect deployment apply` —> [`prefect deploy`](https://docs.prefect.io/v3/deploy/infrastructure-concepts/prefect-yaml#deployment-declaration-reference) Prefect now automatically detects flows in your repo and provides a [wizard](https://docs.prefect.io/v3#step-5-deploy-the-flow) to guide you through setting required attributes for your deployments. `Deployment.build_from_flow` —> [`flow.deploy`](https://reference.prefect.io/prefect/flows/#prefect.flows.Flow.deploy)
  2. **Configuring remote flow code storage:** storage blocks —> [pull action](https://docs.prefect.io/v3/deploy/infrastructure-concepts/prefect-yaml#the-pull-action) When using the YAML-based deployment API, you can configure a pull action in your `prefect.yaml` file to specify how to retrieve flow code for your deployments. You can use configuration from your existing storage blocks to define your pull action [through templating](https://docs.prefect.io/v3/deploy/infrastructure-concepts/prefect-yaml#templating-options). When using the Python deployment API, you can pass any storage block to the `flow.deploy` method to specify how to retrieve flow code for your deployment.
  3. **Configuring flow run infrastructure:** infrastructure blocks —> [typed work pool](https://docs.prefect.io/v3/deploy/infrastructure-concepts/workers#worker-types) Default infrastructure config is now set on the typed work pool, and can be overwritten by individual deployments.
  4. **Managing multiple deployments:** Create and/or update many deployments at once through a [`prefect.yaml`](https://docs.prefect.io/v3/deploy/infrastructure-concepts/prefect-yaml#work-with-multiple-deployments-with-prefect-yaml) file or use the [`deploy`](https://docs.prefect.io/v3/how-to-guides/deployments/deploy-via-python) function.


## 
[​](https://docs.prefect.io/v3/how-to-guides/migrate/upgrade-agents-to-workers#what%E2%80%99s-similar)
What’s similar
  * You can set storage blocks as the pull action in a `prefect.yaml` file.
  * Infrastructure blocks have configuration fields similar to typed work pools.
  * Deployment-level infrastructure overrides operate in much the same way. `infra_override` -> [`job_variable`](https://docs.prefect.io/v3/deploy/infrastructure-concepts/prefect-yaml#work-pool-fields)
  * The process for starting an agent and [starting a worker](https://docs.prefect.io/v3/deploy/infrastructure-concepts/workers#start-a-worker) in your environment are virtually identical. `prefect agent start --pool <work pool name>` —> `prefect worker start --pool <work pool name>`


**Worker Helm chart** If you host your agents in a Kubernetes cluster, you can use the [Prefect worker Helm chart](https://github.com/PrefectHQ/prefect-helm/tree/main/charts/prefect-worker) to host workers in your cluster.
## 
[​](https://docs.prefect.io/v3/how-to-guides/migrate/upgrade-agents-to-workers#upgrade-steps)
Upgrade steps
If you have existing deployments that use infrastructure blocks, you can quickly upgrade them to be compatible with workers by following these steps:
  1. **[Create a work pool](https://docs.prefect.io/v3/deploy/infrastructure-concepts/work-pools#work-pool-configuration)**

This new work pool replaces your infrastructure block. You can use the [`.publish_as_work_pool`](https://docs.prefect.io/2.19.2/api-ref/prefect/infrastructure/#prefect.infrastructure.Infrastructure.publish_as_work_pool) method on any infrastructure block to create a work pool with the same configuration. For example, if you have a `KubernetesJob` infrastructure block named ‘my-k8s-job’, you can create a work pool with the same configuration with this script:
Copy
```
from prefect.infrastructure import KubernetesJob
KubernetesJob.load("my-k8s-job").publish_as_work_pool()

```

Running this script creates a work pool named ‘my-k8s-job’ with the same configuration as your infrastructure block.
**Serving flows** If you are using a `Process` infrastructure block and a `LocalFilesystem` storage block (or aren’t using an infrastructure and storage block at all), you can use [`flow.serve`](https://docs.prefect.io/v3/deploy/index) to create a deployment without specifying a work pool name or start a worker.This is a quick way to create a deployment for a flow and manage your deployments if you don’t need the dynamic infrastructure creation or configuration offered by workers.
  1. **[Start a worker](https://docs.prefect.io/v3/deploy/infrastructure-concepts/workers#start-a-worker)**

This worker replaces your agent and polls your new work pool for flow runs to execute.
Copy
```
prefect worker start -p <work pool name>

```

  1. **Deploy your flows to the new work pool**

To deploy your flows to the new work pool, use `flow.deploy` for a Pythonic deployment experience or `prefect deploy` for a YAML-based deployment experience. If you currently use `Deployment.build_from_flow`, we recommend using `flow.deploy`. If you currently use `prefect deployment build` and `prefect deployment apply`, we recommend using `prefect deploy`.
### 
[​](https://docs.prefect.io/v3/how-to-guides/migrate/upgrade-agents-to-workers#use-flow-deploy)
Use `flow.deploy`
If you have a Python script that uses `Deployment.build_from_flow` to create a deployment, you can replace it with `flow.deploy`. You can translate most arguments to `Deployment.build_from_flow` directly to `flow.deploy`, but here are some possible changes you may need:
  * Replace `infrastructure` with `work_pool_name`. 
    * If you’ve used the `.publish_as_work_pool` method on your infrastructure block, use the name of the created work pool.
  * Replace `infra_overrides` with `job_variables`.
  * Replace `storage` with a call to [`flow.from_source`](https://docs.prefect.io/v3/deploy/index). 
    * `flow.from_source` loads your flow from a remote storage location and makes it deployable. You can pass your existing storage block to the `source` argument of `flow.from_source`.

Below are some examples of how to translate `Deployment.build_from_flow` into `flow.deploy`.
#### 
[​](https://docs.prefect.io/v3/how-to-guides/migrate/upgrade-agents-to-workers#deploying-from-a-local-file)
Deploying from a local file
Using agents and `Deployment.build_from_flow` to deploy a flow from a local file looked like:
Copy
```
from prefect import flow
@flow(log_prints=True)
def my_flow(name: str = "world"):
    print(f"Hello {name}! I'm a flow from a Python script!")
if __name__ == "__main__":
    Deployment.build_from_flow(
        my_flow,
        name="my-deployment",
        parameters=dict(name="Marvin"),
    )

```

When using workers, you can accomplish the same local-storage deployment with `flow.deploy`:
example.py
Copy
```
from pathlib import Path
from prefect import flow
@flow(log_prints=True)
def my_flow(name: str = "world"):
    print(f"Hello {name}! I'm a flow from a Python script!")
if __name__ == "__main__":
    my_flow.from_source(
        source=str(Path(__file__).parent),
        entrypoint="example.py:my_flow",
    ).deploy(
        name="my-deployment",
        parameters=dict(name="Marvin"),
        work_pool_name="local",
    )

```

You can then start a worker to execute scheduled runs, pulling the flow code from `example.py`:
Copy
```
# starts a worker and creates `local` Process work pool if it doesn't exist
prefect worker start --pool local

```

If you’d like to immediately serve this flow as a deployment without running a worker or using work pools, you can [use `flow.serve`](https://docs.prefect.io/v3/how-to-guides/deployment_infra/run-flows-in-local-processes).
#### 
[​](https://docs.prefect.io/v3/how-to-guides/migrate/upgrade-agents-to-workers#deploying-using-a-storage-block)
Deploying using a storage block
If you currently use a storage block to load your flow code but no infrastructure block:
Copy
```
from prefect import flow
from prefect.filesystems import GitHub
@flow(log_prints=True)
def my_flow(name: str = "world"):
    print(f"Hello {name}! I'm a flow from a GitHub repo!")
if __name__ == "__main__":
    Deployment.build_from_flow(
        my_flow,
        name="my-deployment",
        storage=GitHub.load("demo-repo"),
        parameters=dict(name="Marvin"),
    )

```

You can use `flow.from_source` to load your flow from the same location and `flow.deploy` to create a deployment:
example.py
Copy
```
from prefect import flow
from prefect.blocks.system import Secret
from prefect.runner.storage import GitRepository
@flow(log_prints=True)
def my_flow(name: str = "world"):
    print(f"Hello {name}! I'm a flow from a GitHub repo!")
if __name__ == "__main__":
    flow.from_source(
        source=GitRepository(
            url="https://github.com/me/myrepo.git",
            credentials={"username": "oauth2", "access_token": Secret.load("my-github-pat")},
        ),
        entrypoint="example.py:my_flow"
    ).deploy(
        name="my-deployment",
        parameters=dict(name="Marvin"),
        work_pool_name="local", # or the name of your work pool
    )

```

#### 
[​](https://docs.prefect.io/v3/how-to-guides/migrate/upgrade-agents-to-workers#deploy-using-an-infrastructure-block-and-a-storage-block)
Deploy using an infrastructure block and a storage block
For the code below, you need to create a work pool from your infrastructure block and pass it to `flow.deploy` as the `work_pool_name` argument. You also need to pass your storage block to `flow.from_source` as the `source` argument.
example.py
Copy
```
from prefect import flow
from prefect.deployments import Deployment
from prefect.filesystems import GitHub # this block class no longer exists
from prefect.infrastructure.kubernetes import KubernetesJob
@flow(log_prints=True)
def my_flow(name: str = "world"):
    print(f"Hello {name}! I'm a flow from a GitHub repo!")
repo = GitHub.load("demo-repo")
if __name__ == "__main__":
    Deployment.build_from_flow(
        my_flow,
        name="my-deployment",
        storage=repo,
        entrypoint="example.py:my_flow",
        infrastructure=KubernetesJob.load("my-k8s-job"),
        infra_overrides=dict(pull_policy="Never"),
        parameters=dict(name="Marvin"),
    )

```

The equivalent deployment code using `flow.deploy` should look like this:
example.py
Copy
```
from prefect import flow
if __name__ == "__main__":
    flow.from_source(
        source="https://github.com/me/myrepo.git",
        entrypoint="example.py:my_flow"
    ).deploy(
        name="my-deployment",
        work_pool_name="my-k8s-job",
        job_variables=dict(pull_policy="Never"),
        parameters=dict(name="Marvin"),
    )

```

When using `flow.from_source().deploy()` with a remote `source`such as a `GitHub` block or `str` URL like <https://github.com/me/myrepo.git>), the flow you’re deploying doesn’t need to be available locally before running your script. See the [SDK reference](https://reference.prefect.io/prefect/#prefect.Flow.from_source) for more info on `from_source`.
#### 
[​](https://docs.prefect.io/v3/how-to-guides/migrate/upgrade-agents-to-workers#deploy-via-a-docker-image)
Deploy via a Docker image
If you currently bake your flow code into a Docker image before deploying, you can use the `image` argument of `flow.deploy` to build a Docker image as part of your deployment process:
Copy
```
from prefect import flow
@flow(log_prints=True)
def my_flow(name: str = "world"):
    print(f"Hello {name}! I'm a flow from a Docker image!")
if __name__ == "__main__":
    my_flow.deploy(
        name="my-deployment",
        image="my-repo/my-image:latest",
        work_pool_name="my-k8s-job",
        job_variables=dict(pull_policy="Never"),
        parameters=dict(name="Marvin"),
    )

```

You can skip a `flow.from_source` call when building an image with `flow.deploy`. Prefect keeps track of the flow’s source code location in the image and loads it from that location when the flow is executed.
### 
[​](https://docs.prefect.io/v3/how-to-guides/migrate/upgrade-agents-to-workers#use-prefect-deploy)
Use `prefect deploy`
**Always run`prefect deploy` commands from the `root` level of your repo!**With agents, you may have multiple `deployment.yaml` files. But under worker deployment patterns, each repo has a single `prefect.yaml` file located at the **root** of the repo that contains [deployment configuration](https://docs.prefect.io/v3/deploy/infrastructure-concepts/prefect-yaml#work-with-multiple-deployments-with-prefect-yaml) for all flows in that repo.
To set up a new `prefect.yaml` file for your deployments, run the following command from the root level of your repo:
Copy
```
prefect deploy

```

This starts a wizard that guides you through setting up your deployment.
**For step 4, select`y` on the last prompt to save the configuration for the deployment.**Saving the configuration for your deployment results in a `prefect.yaml` file populated with your first deployment. You can use this YAML file to edit and [define multiple deployments](https://docs.prefect.io/v3/deploy/infrastructure-concepts/prefect-yaml#work-with-multiple-deployments-with-prefect-yaml) for this repo.
You can add more [deployments](https://docs.prefect.io/v3/deploy/infrastructure-concepts/prefect-yaml#deployment-declaration-reference) to the `deployments` list in your `prefect.yaml` file and/or by continuing to use the deployment creation wizard. For more information on deployments, check out our [in-depth guide for deploying flows to work pools](https://docs.prefect.io/v3/how-to-guides/deployment_infra/serve-flows-docker).
Was this page helpful?
YesNo
[Upgrade to Prefect 3.0](https://docs.prefect.io/v3/how-to-guides/migrate/upgrade-to-prefect-3)[Transfer resources between environments](https://docs.prefect.io/v3/how-to-guides/migrate/transfer-resources)
