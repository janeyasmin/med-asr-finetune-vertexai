import google.cloud.aiplatform as aip
import os
from dotenv import load_dotenv
from datetime import datetime


# Load environment variables from .env file
load_dotenv()

# Setup
HF_TOKEN = os.getenv("HF_TOKEN")
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
REGION = os.getenv("GOOGLE_CLOUD_LOCATION")
BUCKET_URI = os.getenv("GCS_BUCKET_URI")
TRAINING_IMAGE_URI = os.getenv("TRAINING_IMAGE_URI")
SERVING_IMAGE_URI = os.getenv("SERVING_IMAGE_URI")
NUM_TRAIN_EPOCHS = os.getenv("NUM_TRAIN_EPOCHS")
MAX_WER_DEPLOYMENT = os.getenv("MAX_WER_DEPLOYMENT")


# Initialize Vertex AI SDK

aip.init(
    project=PROJECT_ID,
    location=REGION,
    staging_bucket=BUCKET_URI,
)

# Prepare the pipeline job
job = aip.PipelineJob(
    display_name="med-asr-finetune-pipeline-job",
    template_path="med-asr-finetune-pipeline.json",
    parameter_values={
        "project_id": PROJECT_ID,
        "region": REGION,
        "bucket_uri": BUCKET_URI,
        "base_output_dir": f"med-asr-finetune-pipeline-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "hf_token": HF_TOKEN,
        "epochs": NUM_TRAIN_EPOCHS,
        "training_container_image_uri": TRAINING_IMAGE_URI,
        "serving_container_image_uri": SERVING_IMAGE_URI,
        "max_wer_deployment": MAX_WER_DEPLOYMENT,
    },
    enable_caching=False,
)
job.submit()
