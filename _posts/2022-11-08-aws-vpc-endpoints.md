---
layout: single
title:  Using AWS Lambda functions in VPC with VPC endpoints
date: 2022-11-08
mathjax: false
toc: true
toc_sticky: true
tags:
  - Deployment
---

AWS Lambda is a great way to create a serverless event-driven compute service. There are a few set-ups headaches when it comes to having the Lambda function interact with the various AWS services.

The AWS Lambda function can be deployed inside a Virtual Private Cloud (VPC). This is useful when there are services also running in the VPC that the Lambda function needs (e.g. a RDS database, an API running on an EC2 instance). Setting up VPC endpoints is the easiest way to allow the Lambda function access to AWS services (or services on AWS marketplace) without having to route traffic out of VPC (via NAT gateways, etc).

Below is an example of setting up endpoints for access to DynamoDB and Simple Email Service (SES).

## Create DynamoDB VPC gateway endpoint
Go to *Virtual private cloud -> Endpoints -> Create endpoint*. Select *AWS services*, select *com.amazonaws.<region>.dynamodb*. And then select the VPC, Route table, and policy (either full access or a custom policy).

![](/images/aws_vpc_endpoints.webp){: .align-center}

In AWS Identity and Access Management (IAM), make sure the role for the Lambda function also has rights to use the dynamodb API.

## Create SES SMTP VPC interface endpoint

To use the Simple Email Service (SES) to send emails using AWS, first set up the Identities on AWS. In order to use SES to send emails from Lambda, you can generally just use the SES API. However, when Lambda is inside a VPC, you also need to create an endpoint to access the service. At the time of writing, AWS does not support VPC endpoints for AWS SES API. However you can create a SES SMTP endpoint and instead of using the SES API, you can use a SMTP client library (for python, e.g. smtplib).

To create the SES SMTP endpoint, Go to *Virtual private cloud -> Endpoints -> Create endpoint*. Select *AWS services*, select *com.amazonaws.<region>.email-smtp*. And then select the VPC, Subsets, and Security group.

To send the email in python, the code would look something like the following (example with placeholders),

```python
import smtplib
server = "email-smtp.<region>.amazonaws.com"
user = "USERNAME"
password = "PASSWORD"
message = """From: from@domain.com
To: to@domain.com
Subject: subject
email message
"""
try:
   conn = smtplib.SMTP(server)
   conn.starttls()
   conn.login(user, password)
   conn.sendmail(from@domain.com, [to@domain.com], message)
except smtplib.SMTPException as e:
   print(e)
```

With that, the Lambda function now can access to the resources in VPC as well as AWS services via VPC endpoints!
