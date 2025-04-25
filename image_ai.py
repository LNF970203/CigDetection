import base64
from pathlib import Path
from PIL import Image
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import os
from io import BytesIO
import streamlit as st


os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


PROMPT = """
Evaluate the provided images for cigarette pack arrangement:

Reference Image: Displays the correct order as follows:

1. First Row: 2 Dunhill Blue packs followed by 1 Dunhill Tube pack.
2. Second Row: 3 John Player Gold Leaf packs.
3. All packs must be facing upright with letters upright, and no packs should be missing.

Input Image: Verify whether the arrangement matches the reference image's correct order.

Provide an analysis by stating if the order in the input image is correct. 

If the order is incorrect, explain the discrepancies such as the incorrect order of packs, missing packs, or incorrect pack orientation, specifying which parts of the input image do not comply with the reference order.

"""


# set the openai model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def load_as_data_uri(img_path: str) -> str:
    """Open an image, encode it as base-64 data-URI string (PNG)."""
    img = Image.open(img_path).convert("RGB")
    # re-encode as PNG so every format works
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def compare_images(reference_uri: str, input_uri: str):
    reference_image = load_as_data_uri(reference_uri)
    input_image     = load_as_data_uri(input_uri)

    prompt = [
        {
            "type": "text",
            "text": (
                PROMPT
            ),
        },
        {
            "type": "image_url",
            "image_url": {"url": reference_image},
        },
        {
            "type": "image_url",
            "image_url": {"url": input_image},
        },
    ]

    response = llm([HumanMessage(content=prompt)])

    return response.content