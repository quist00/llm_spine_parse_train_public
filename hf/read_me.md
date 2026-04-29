1. Create a Hugging Face account (if you don’t have one)
    - Go to https://huggingface.co/join
2. Create a Write token  
- Go to https://huggingface.co/settings/tokens  
- Click “New token” → Role = Write → Name it something like “book-spine-upload”  
- Copy the token (it starts with hf_)

3. Install the required packages (run once)
    pip install -U ultralytics huggingface_hub peft transformers gradio

4. Log into hugging face
    huggingface-cli login
- Paste token

5. use appropriate py file to upload.