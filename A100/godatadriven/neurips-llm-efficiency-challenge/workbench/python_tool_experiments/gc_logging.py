# %%
import logging
from google.cloud.logging import Client

# %%

from cloud_detect import provider

provider()

# %%

client = Client()

# %%

client.setup_logging()

# %%

logging.info("test (root)")

# %%

logger = logging.getLogger("workbench")

# %%

logger.info("test (workbench)")

# %%


class myFilter(logging.Filter):
    def filter(self, record):
        labels = {}
        labels["boo"] = "baa"
        record.labels = labels
        return True


# %%

logger.addFilter(myFilter())


# %%

logger.info("Test (workbench /w filter)")

# %%
