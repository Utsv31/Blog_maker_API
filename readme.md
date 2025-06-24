📝 AWS Bedrock Blog Maker API
📖 Overview
This project is a serverless API that:

Uses AWS Bedrock's LLaMA3-70b-instruct model to generate a 200-word blog on any topic.

Saves the generated blog into an S3 bucket as a .txt file.

Runs on AWS Lambda and is triggered via API Gateway.

Testable with tools like Postman.

🧠 Workflow
API Gateway receives a POST request containing the blog_topic.

Lambda Handler invokes the Bedrock model (meta.llama3-70b-instruct-v1:0) to generate a blog.

Generated blog is uploaded as a timestamped file into the specified S3 bucket.

Returns a 200 response indicating successful completion.

🧰 Tech Stack
AWS Lambda (Python)

AWS API Gateway (REST API trigger)

AWS S3 (to store generated blogs)

AWS Bedrock Runtime (LLaMA model)

Postman (for testing)

input:
{
    "blog_topic":"Machine Learning and Artificial Intelligence"
}

 Example Output:
 "Blog Generation is completed"

 Configuration
Before deploying, make sure to:

Set up an S3 bucket (s3_bucket='awsbucketbedrock310505' in the code) in the same region.

Attach the proper IAM roles for Lambda with permissions to:

Invoke bedrock:InvokeModel

s3:PutObject into your bucket

Deploy the API Gateway and note its invoke URL.