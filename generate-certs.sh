#!/bin/bash
# This script generates self-signed certificates for local HTTPS development

# Create certificates directory if it doesn't exist
mkdir -p ./server/certificates

# Generate a private key
openssl genrsa -out ./server/certificates/key.pem 2048

# Generate a certificate signing request (CSR)
openssl req -new -key ./server/certificates/key.pem -out ./server/certificates/csr.pem -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"

# Generate a self-signed certificate (valid for 365 days)
openssl x509 -req -days 365 -in ./server/certificates/csr.pem -signkey ./server/certificates/key.pem -out ./server/certificates/cert.pem

# Remove the CSR as it's no longer needed
rm ./server/certificates/csr.pem

echo "Self-signed certificates generated successfully in ./server/certificates/"
