# Deployment Guide - Lovdata Legal AI

This guide provides multiple options for deploying the Lovdata Legal AI application permanently.

## Quick Deploy Options

### Option 1: Railway (Recommended - Free Tier Available)

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template/new)

**Steps:**
1. Click the button above
2. Connect your GitHub account
3. Select the `lovdata-legal-ai-space` repository
4. Railway will automatically detect the Dockerfile and deploy
5. Your app will be live at `https://your-app.railway.app`

**Configuration:**
- No configuration needed
- Automatic HTTPS
- Free tier: 500 hours/month

### Option 2: Render

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

**Steps:**
1. Click the button above
2. Connect your GitHub repository: `https://github.com/Jakobkoding2/lovdata-legal-ai-space`
3. Render will auto-detect the Dockerfile
4. Click "Create Web Service"
5. Your app will be live at `https://your-app.onrender.com`

**Configuration:**
- Instance Type: Free (or upgrade for better performance)
- Auto-deploy: Enabled

### Option 3: Fly.io

**Steps:**
1. Install Fly CLI: `curl -L https://fly.io/install.sh | sh`
2. Login: `flyctl auth login`
3. Clone the repo:
   ```bash
   git clone https://github.com/Jakobkoding2/lovdata-legal-ai-space.git
   cd lovdata-legal-ai-space
   ```
4. Launch:
   ```bash
   flyctl launch
   ```
5. Deploy:
   ```bash
   flyctl deploy
   ```

**Configuration:**
- Free tier: 3 shared-cpu-1x 256MB VMs
- Automatic HTTPS
- Global CDN

### Option 4: Google Cloud Run

**Steps:**
1. Install gcloud CLI
2. Clone the repository
3. Build and deploy:
   ```bash
   gcloud run deploy lovdata-legal-ai \
     --source . \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated
   ```

**Configuration:**
- Pay-per-use pricing
- Automatic scaling
- Free tier: 2 million requests/month

### Option 5: Hugging Face Spaces

**Steps:**
1. Go to [https://huggingface.co/new-space](https://huggingface.co/new-space)
2. Choose "Gradio" as the SDK
3. Connect this GitHub repository or upload files manually
4. Your space will be live at `https://huggingface.co/spaces/YOUR_USERNAME/lovdata-legal-ai`

**Configuration:**
- Free tier available
- Automatic HTTPS
- Community features

### Option 6: Docker (Self-Hosted)

**Prerequisites:**
- Docker installed
- Docker Compose installed

**Steps:**
1. Clone the repository:
   ```bash
   git clone https://github.com/Jakobkoding2/lovdata-legal-ai-space.git
   cd lovdata-legal-ai-space
   ```

2. Build and run:
   ```bash
   docker-compose up -d
   ```

3. Access at `http://localhost:7860`

**For production:**
- Use a reverse proxy (nginx/Caddy) for HTTPS
- Set up a domain name
- Configure firewall rules

## Environment Variables

No environment variables are required for basic operation. Optional:

- `GRADIO_SERVER_NAME`: Server host (default: 0.0.0.0)
- `GRADIO_SERVER_PORT`: Server port (default: 7860)

## Resource Requirements

**Minimum:**
- RAM: 2GB
- CPU: 1 core
- Disk: 500MB

**Recommended:**
- RAM: 4GB
- CPU: 2 cores
- Disk: 1GB

## Monitoring

Once deployed, monitor your application:

1. **Health Check**: Visit `https://your-app-url/` to verify it's running
2. **Logs**: Check platform-specific logs for errors
3. **Performance**: Monitor response times in the Gradio interface

## Troubleshooting

### App won't start
- Check logs for errors
- Verify sufficient memory (2GB minimum)
- Ensure all files are uploaded correctly

### Slow performance
- Upgrade to a larger instance
- Enable caching
- Consider using a GPU instance for faster embeddings

### Out of memory
- Reduce the corpus size in `app.py`
- Upgrade to an instance with more RAM
- Enable swap memory

## Cost Estimates

| Platform | Free Tier | Paid Tier (Basic) |
|----------|-----------|-------------------|
| Railway | 500 hrs/month | $5/month |
| Render | 750 hrs/month | $7/month |
| Fly.io | 3 VMs free | $1.94/month |
| Google Cloud Run | 2M requests/month | Pay-per-use |
| Hugging Face | Unlimited (CPU) | $9/month (GPU) |

## Support

For deployment issues:
- GitHub Issues: [lovdata-legal-ai-space/issues](https://github.com/Jakobkoding2/lovdata-legal-ai-space/issues)
- Original Repo: [lovdata-legal-ai](https://github.com/Jakobkoding2/lovdata-legal-ai)

## Security Notes

- This is a demonstration system
- Do not use for production legal advice without expert review
- Consider adding authentication for production use
- Regular security updates recommended
