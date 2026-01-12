from kfp import dsl, compiler
from google_cloud_pipeline_components.v1.custom_job import CustomTrainingJobOp
import argparse

# Setup
BASE_IMAGE = "python:3.12"


@dsl.component(
    packages_to_install=["google-cloud-storage"],
    base_image=BASE_IMAGE,
)
def validate_model_op(
    project_id: str,
    bucket_uri: str,
    base_output_dir: str,
    max_wer: float,
) -> str:
    import json
    from google.cloud import storage

    storage_client = storage.Client(project=project_id)

    bucket_name = bucket_uri.replace("gs://", "")
    bucket = storage_client.bucket(bucket_name)

    eval_json_blob = bucket.blob(f"{base_output_dir}/model/all_results.json")
    eval_json = eval_json_blob.download_as_bytes().decode("utf-8")
    eval_metrics = json.loads(eval_json)
    eval_wer = eval_metrics.get("eval_wer", 100.0)

    print("*** Evaluation Metrics ***")
    for key, value in eval_metrics.items():
        print(f"{key}: {value}")

    if eval_wer <= max_wer:
        return "valid"
    else:
        return "invalid"


@dsl.component(
    packages_to_install=["google-cloud-aiplatform"],
    base_image=BASE_IMAGE,
)
def deploy_model_op(
    project_id: str,
    region: str,
    bucket_uri: str,
    base_output_dir: str,
    model_name: str,
    hf_token: str,
    serving_container_image_uri: str,
):
    from google.cloud import aiplatform

    aiplatform.init(project=project_id, location=region, staging_bucket=bucket_uri)

    artifact_dir = f"{base_output_dir}/model"
    artifact_uri = f"{bucket_uri}/{artifact_dir}"

    matches = aiplatform.Model.list(filter=f"display_name={model_name}")
    parent_model = matches[0].resource_name if matches else None

    print("Uploading model to Vertex AI Model Registry...")
    # Import model to Vertex AI Model Registry
    model = aiplatform.Model.upload(
        display_name=model_name,
        artifact_uri=artifact_uri,
        serving_container_image_uri=serving_container_image_uri,
        serving_container_health_route="/h",
        serving_container_predict_route="/v1/audio/transcriptions",
        serving_container_ports=[8080],
        serving_container_environment_variables={
            "HF_TOKEN": hf_token,
        },
        parent_model=parent_model,
    )

    endpoint_name = f"{model_name}-endpoint"

    endpoints = aiplatform.Endpoint.list(filter=f"display_name={endpoint_name}")
    if endpoints:
        endpoint = endpoints[0]
    else:
        print("Creating new endpoint...")
        endpoint = aiplatform.Endpoint.create(display_name=endpoint_name)

    print("Deploying model to endpoint...")
    model.deploy(endpoint=endpoint, traffic_percentage=100)  # type: ignore


@dsl.pipeline(
    name="med-asr-finetune-pipeline",
    description="A pipeline to fine-tune the MedASR model on the Eka dataset using Vertex AI",
)
def med_asr_finetune_pipeline(
    project_id: str,
    region: str,
    bucket_uri: str,
    base_output_dir: str,
    hf_token: str,
    epochs: int,
    training_container_image_uri: str,
    serving_container_image_uri: str,
    max_wer_deployment: float,
):

    model_name = "med-asr-finetuned-model"

    training_task = CustomTrainingJobOp(
        display_name="med-asr-finetune-training-job",
        location=region,
        worker_pool_specs=[
            {
                "machineSpec": {
                    "machineType": "n1-standard-16",
                    "acceleratorType": "NVIDIA_TESLA_T4",
                    "acceleratorCount": 1,
                },
                "replicaCount": 1,
                "diskSpec": {"bootDiskSizeGb": 200},
                "containerSpec": {
                    "imageUri": training_container_image_uri,
                    "args": [
                        "--batch_size=8",
                        f"--epochs={epochs}",
                        "--do_train",
                    ],
                    "env": [
                        {"name": "HF_TOKEN", "value": hf_token},
                    ],
                },
            }
        ],
        base_output_directory=f"{bucket_uri}/{base_output_dir}",
    ).set_display_name("train-model")

    evaluate_task = CustomTrainingJobOp(
        display_name="med-asr-finetune-evaluation-job",
        location=region,
        worker_pool_specs=[
            {
                "machineSpec": {
                    "machineType": "n1-standard-16",
                    "acceleratorType": "NVIDIA_TESLA_T4",
                    "acceleratorCount": 1,
                },
                "replicaCount": 1,
                "diskSpec": {"bootDiskSizeGb": 200},
                "containerSpec": {
                    "imageUri": training_container_image_uri,
                    "args": [
                        "--do_eval",
                    ],
                    "env": [
                        {"name": "HF_TOKEN", "value": hf_token},
                    ],
                },
            }
        ],
        base_output_directory=f"{bucket_uri}/{base_output_dir}",
    ).set_display_name("evaluate-model")
    evaluate_task.after(training_task)  # type: ignore

    validate_task = validate_model_op(
        project_id=project_id,
        bucket_uri=bucket_uri,
        base_output_dir=base_output_dir,
        max_wer=max_wer_deployment,
    ).set_display_name(  # pyright: ignore[reportAttributeAccessIssue]
        "validate-model-for-upload"
    )

    validate_task.after(evaluate_task)  # type: ignore

    with dsl.If(
        validate_task.output == "valid",  # type: ignore
        name="check-wer-less-than-max",
    ):
        deploy_task = deploy_model_op(
            project_id=project_id,
            region=region,
            bucket_uri=bucket_uri,
            base_output_dir=base_output_dir,
            model_name=model_name,
            hf_token=hf_token,
            serving_container_image_uri=serving_container_image_uri,
        )
        deploy_task.set_display_name("deploy-model")  # type: ignore


def compile(filename: str):
    compiler.Compiler().compile(
        pipeline_func=med_asr_finetune_pipeline,  # type: ignore
        package_path=filename,
    )
    print(f"Pipeline compiled to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pipeline-file-name",
        type=str,
        default="med-asr-finetune-pipeline.json",
    )

    args = parser.parse_args()

    compile(args.pipeline_file_name)
