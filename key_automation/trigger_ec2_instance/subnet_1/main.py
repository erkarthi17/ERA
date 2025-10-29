import boto3, time, smtplib, traceback
from email.mime.text import MIMEText
from datetime import datetime, timedelta
import os
import dotenv
import pytz
import dotenv

dotenv.load_dotenv()

# AWS credentials and region
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_REGION = "eu-west-2"

# Email configuration (for notifications)
SENDER_EMAIL = "erkarthi17@gmail.com"
RECEIVER_EMAIL = "erkarthi17@gmail.com"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USERNAME = os.getenv("SMTP_USERNAME")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")

# EC2 instance configuration
AMI_ID = os.getenv("AMI_ID")  
INSTANCE_TYPE = os.getenv("INSTANCE_TYPE") 
KEY_NAME = os.getenv("KEY_NAME") 
# SECURITY_GROUP_ID = os.getenv("SECURITY_GROUP_ID") 
SECURITY_GROUP_ID = ['sg-0aa02f02b2cf95dca']
SUBNET_ID = os.getenv("SUBNET_ID")
INSTANCE_TAGS = [
    {"Key": "Name", "Value": "Image_Net_Mini_Capstone"},
    {"Key": "Project", "Value": "ERA"},
]

def send_email_notification(subject, body):
    """Sends an email notification."""
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = SENDER_EMAIL
        msg["To"] = RECEIVER_EMAIL

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)
        print("Email notification sent successfully.")
    except Exception as e:
        print(f"Failed to send email notification: {e}")

def get_running_instances():
    """Returns a list of currently running EC2 instances."""
    try:
        ec2 = boto3.client(
            "ec2",
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        )
        response = ec2.describe_instances(
            Filters=[
                {"Name": "instance-state-name", "Values": ["running", "pending"]}
            ]
        )
        running_instances = []
        for reservation in response["Reservations"]:
            for instance in reservation["Instances"]:
                running_instances.append(instance["InstanceId"])
        return running_instances
    except Exception as e:
        print(f"⚠️ Error checking existing instances: {e}")
        return []

def launch_instance():
    """Launches a new EC2 Spot Instance."""
    try:
        ec2 = boto3.client(
            "ec2",
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        )

        # Specify the latest Ubuntu AMI for us-east-1 (example)
        # You might want to automate finding the latest AMI or use a fixed one.
        # This is a placeholder; get the actual AMI ID from AWS console or API.
        ami_id = AMI_ID
        instance_type = INSTANCE_TYPE
        key_name = KEY_NAME
        security_group_id = SECURITY_GROUP_ID
        subnet_id = SUBNET_ID

        print(f"Attempting to launch instance with AMI: {ami_id}, Type: {instance_type}")

        # Launch the EC2 instance
        response = ec2.run_instances(
            ImageId=ami_id,
            MinCount=1,
            MaxCount=1,
            InstanceType=instance_type,
            KeyName=key_name,
            NetworkInterfaces=[
                {
                    "DeviceIndex": 0,
                    "Groups": security_group_id,
                    "SubnetId": subnet_id
                }
            ],
            TagSpecifications=[
                {
                    "ResourceType": "instance",
                    "Tags": INSTANCE_TAGS,
                },
            ],
            InstanceInitiatedShutdownBehavior="terminate",  # Terminate on shutdown
        )

        instance_id = response["Instances"][0]["InstanceId"]
        print(f"🚀 Successfully launched EC2 instance: {instance_id}")

        # Log the launch event
        with open("spot_launch.log", "a", encoding="utf-8") as f:
            f.write(
                f"{datetime.now(pytz.utc).isoformat()} - Launched instance {instance_id}\n"
            )

        send_email_notification(
            "EC2 Instance Launched", f"EC2 instance {instance_id} has been launched."
        )
        return True

    except Exception as e:
        error_message = f"❌ Unexpected error: {traceback.format_exc()}"
        print(error_message)
        # Log the error
        fallback_message = f"An error occurred at {datetime.now(pytz.utc).isoformat()}: {e}\n"
        with open("spot_launch.log", "a", encoding="utf-8") as f:
            f.write(fallback_message)
        send_email_notification("EC2 Launch Failed", error_message)
        return False

def check_and_launch():
    """Checks for running instances and launches one if none are active."""
    print("Checking for running instances...")
    running_instances = get_running_instances()

    if not running_instances:
        print("No running instances found. Launching a new one...")
        success = launch_instance()
        if success:
            print("Instance launch initiated successfully.")
        else:
            print("Failed to initiate instance launch.")
    else:
        print(f"Found {len(running_instances)} running instance(s): {running_instances}. No new instance will be launched.")

if __name__ == "__main__":
    success = launch_instance()