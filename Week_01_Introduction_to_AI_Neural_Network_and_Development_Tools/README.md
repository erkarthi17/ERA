## Setup

1. Install dependencies:
    ```
    pip install -r requirements.txt
    ```

2. Create a `.env` file with:
    ```
    CANVAS_API_URL=your_canvas_api_url
    CANVAS_API_TOKEN=your_canvas_api_token
    ```

3. Run the backend:
    ```
    uvicorn app:app --reload
    ```

4. Load the extension in Chrome and use it.