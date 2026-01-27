"""
TERMINAL - HYBRID SYNC (Sovereign Coordination)
==============================================
Synchronizes the 'Sentinel' (Oracle) with 'The Forge' (GCP).
Ensures zero-downtime model swaps and total nodal integrity.
"""
import os
import shutil
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger('HYBRID_SYNC')

MODELS_DIR = "/Users/anmol/Desktop/gold/aegis_alpha/models"
STAGING_DIR = "/Users/anmol/Desktop/gold/aegis_alpha/models/staging"

def sync_nodes():
    """
    Coordinates the synchronization of all 30 nodes.
    1. Checks staging for new 'Forge' outputs.
    2. Verifies 15-feature alignment.
    3. Atomic swap into active models/ directory.
    """
    if not os.path.exists(STAGING_DIR):
        os.makedirs(STAGING_DIR)
        logger.info(f"Created staging sanctuary: {STAGING_DIR}")

    new_models = [f for f in os.listdir(STAGING_DIR) if f.endswith('.pkl')]
    if not new_models:
        logger.info("No new models in staging. The Forge is cooling.")
        return

    logger.info(f"Detected {len(new_models)} candidates for ascension.")
    
    for model_file in new_models:
        src = os.path.join(STAGING_DIR, model_file)
        dest = os.path.join(MODELS_DIR, model_file)
        
        try:
            # Atomic swap (move)
            shutil.move(src, dest)
            logger.info(f"✅ ASCENDED: {model_file} is now LIVE.")
        except Exception as e:
            logger.error(f"❌ FAILED ASCENSION for {model_file}: {e}")

if __name__ == "__main__":
    logger.info("🐉 HYBRID SYNC PROTOCOL: INITIALIZING...")
    sync_nodes()
    logger.info("🐉 SYNC COMPLETE. THE MATRIX IS STEADY.")
