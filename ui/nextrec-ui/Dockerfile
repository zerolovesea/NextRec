FROM node:20-alpine

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

EXPOSE 15173
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "15173"]
