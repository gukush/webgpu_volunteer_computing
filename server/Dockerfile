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

# This command will copy package.json.
# If you happen to have a package-lock.json in your build context, it will also be copied,
# and "npm install" will use it. Otherwise, "npm install" will create one inside the image.
COPY package*.json ./

# Removed "RUN npm ci --omit=dev --verbose"
# Now, npm install will handle the installation.
# It will generate a package-lock.json inside the image if one wasn't copied.
# --omit=dev ensures devDependencies (like nodemon) are not installed in the final image.
RUN npm install --omit=dev --verbose

# Copy the rest of your application code
COPY . .

EXPOSE 3000

# Puppeteer configuration:
# The system dependencies installed via apt-get should be sufficient for Puppeteer.
# PUPPETEER_SKIP_CHROMIUM_DOWNLOAD=true tells Puppeteer not to download its own Chromium.
ENV PUPPETEER_SKIP_CHROMIUM_DOWNLOAD=true
# You might need to tell Puppeteer where to find the system's Chromium if it doesn't auto-detect.
# Common paths are /usr/bin/chromium or /usr/bin/chromium-browser.
# If Puppeteer can't find Chromium, you might need to set:
# ENV PUPPETEER_EXECUTABLE_PATH=/usr/bin/chromium-browser (or the correct path)

CMD ["node", "server.js"]