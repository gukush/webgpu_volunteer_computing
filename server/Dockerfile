FROM node:18-slim

# Install system deps required by Chromium for Puppeteer:
RUN apt-get update && apt-get install -y \
      wget gnupg ca-certificates fonts-liberation \
      libatk1.0-0 libatk-bridge2.0-0 libcairo2 libpango-1.0-0 \
      libx11-xcb1 libxcomposite1 libxdamage1 libxrandr2 libgbm1 \
      libasound2 libnss3 libxss1 xvfb \
      --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY package*.json ./

# Using npm ci for cleaner installs if package-lock.json is present and up-to-date
# If not, npm install is fine.
RUN npm ci --omit=dev || npm install --omit=dev

COPY . .

EXPOSE 3000

# Set PUPPETEER_SKIP_CHROMIUM_DOWNLOAD to true if you've handled Chromium separately
# ENV PUPPETEER_EXECUTABLE_PATH /usr/bin/google-chrome-stable

CMD ["node", "server.js"]