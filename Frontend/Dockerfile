FROM node:18-alpine AS builder

WORKDIR /app

COPY package.json package-lock.json* ./
# Install dependencies (ignoring scripts to avoid issues, or safely run them)
RUN npm install

COPY . .

ARG VITE_API_URL
ENV VITE_API_URL=$VITE_API_URL

RUN npm run build

FROM nginx:alpine

COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
