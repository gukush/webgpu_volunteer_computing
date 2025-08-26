#!/usr/bin/env node
// test-endpoints.mjs - Test all available endpoints
import https from 'https';

const httpsAgent = new https.Agent({
  rejectUnauthorized: false
});

const BASE_URL = process.argv[2] || 'https://localhost:3000';

async function testEndpoint(method, path, body = null) {
  const url = `${BASE_URL}${path}`;
  console.log(`\n Testing ${method} ${path}`);
  
  const options = {
    method,
    agent: httpsAgent
  };
  
  if (body) {
    options.headers = { 'Content-Type': 'application/json' };
    options.body = JSON.stringify(body);
  }
  
  try {
    const response = await fetch(url, options);
    const text = await response.text();
    
    console.log(`   Status: ${response.status} ${response.statusText}`);
    
    if (response.ok) {
      try {
        const json = JSON.parse(text);
        console.log(`    Response: ${JSON.stringify(json, null, 2)}`);
      } catch {
        console.log(`    Response: ${text.slice(0, 200)}...`);
      }
    } else {
      console.log(`    Error: ${text.slice(0, 200)}...`);
    }
  } catch (error) {
    console.log(`    Request failed: ${error.message}`);
  }
}

async function main() {
  console.log(`Testing endpoints on: ${BASE_URL}`);
  
  // Test basic connectivity
  await testEndpoint('GET', '/api/status');
  
  // Test the problematic endpoint
  await testEndpoint('POST', '/api/workloads/startQueued');
  
  // Test other endpoints for comparison
  await testEndpoint('GET', '/api/workloads');
  await testEndpoint('POST', '/api/system/k', { k: 1 });
  
  console.log('\n Summary:');
  console.log('If /api/status works but /api/workloads/startQueued gives 404,');
  console.log('there might be an issue with route registration in server.js');
}

main().catch(console.error);
