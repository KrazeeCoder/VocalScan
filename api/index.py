from backend.app import create_app

# Vercel Python serverless entrypoint: expose Flask app as `app`
app = create_app()


