import os
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup():
    """
    Clone repository and set up required directories
    """
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('voices', exist_ok=True)
    
    # Install git-lfs if not already installed
    try:
        subprocess.run(['git', 'lfs', 'install'], check=True)
    except subprocess.CalledProcessError:
        logger.error("Failed to install git-lfs. Please install it manually.")
        return
    except FileNotFoundError:
        logger.error("git-lfs not found. Please install git and git-lfs first.")
        return
    
    # Clone the repository if it doesn't exist
    if not os.path.exists('Kokoro-82M'):
        logger.info("Cloning Kokoro-82M repository...")
        try:
            subprocess.run([
                'git', 'clone',
                'https://huggingface.co/hexgrad/Kokoro-82M'
            ], check=True)
            logger.info("Repository cloned successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone repository: {e}")
            return
    else:
        logger.info("Kokoro-82M repository already exists")
    
    # Create symlinks to the required files
    files_to_link = [
        ('kokoro-v0_19.pth', 'models/kokoro-v0_19.pth'),
        ('models.py', 'models.py'),
        ('kokoro.py', 'kokoro.py'),
        ('istftnet.py', 'istftnet.py'),
        ('plbert.py', 'plbert.py'),
        ('config.json', 'config.json')
    ]
    
    for src, dst in files_to_link:
        src_path = os.path.join('Kokoro-82M', src)
        if not os.path.exists(dst):
            try:
                if os.path.exists(src_path):
                    os.symlink(src_path, dst)
                    logger.info(f"Created symlink for {dst}")
                else:
                    logger.error(f"Source file {src_path} not found")
            except OSError as e:
                logger.error(f"Failed to create symlink for {dst}: {e}")
    
    logger.info("Setup complete! You can now run the server.")

if __name__ == "__main__":
    setup() 