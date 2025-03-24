import gradio as gr
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Step 1: Load the dataset
file_path = "/content/SyntheticGrievances.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1")

# Step 2: Generate embeddings for the "Message Description" field
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model.to(device)
descriptions = df["Message Description"].tolist()
embeddings = embedding_model.encode(descriptions)

# Step 3: Store embeddings in a FAISS index for similarity search
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Step 4: Define a function to retrieve similar grievances
def retrieve_similar_grievances(query, top_k=3):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    return df.iloc[indices[0]], distances[0]

# Step 5: Load pre-trained models for sentiment analysis and generation
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",  # Outputs "NEGATIVE"/"POSITIVE"
    device=0 if torch.cuda.is_available() else -1
)

model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

# Step 6: Define a function to generate recommendations
def generate_recommendation(query, context, distances):
    relevant_context = any(distance < 0.7 for distance in distances)  # Adjust threshold if needed
    if relevant_context:
        prompt = (
            f"Problem: {query}\n"
            f"Context: {context}\n"
            f"Task: Suggest a coherent and actionable solution based on the context."
        )
    else:
        # Fallback for lost-item scenarios
        if "wallet" in query.lower() and ("office" in query.lower() or "bus stop" in query.lower()):
            return "Contact the security team at the location immediately, file a report, and check the lost-and-found section."
        else:
            prompt = (
                f"Problem: {query}\n"
                f"Task: Provide a generic but actionable solution."
            )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=150, num_beams=5, repetition_penalty=1.5)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Step 7: End-to-end pipeline
def process_grievance(query):
    similar_grievances, distances = retrieve_similar_grievances(query)
    context = "\n".join(similar_grievances["Message Description"])

    # Analyze sentiment of the query itself (not just retrieved feedback)
    sentiment, score = sentiment_analyzer(query)[0]["label"], sentiment_analyzer(query)[0]["score"]

    # Assign priority based on query sentiment and resolution status
    resolution_status = "Pending"  # Assume new grievances are pending
    priority = assign_priority(sentiment, resolution_status)

    # Generate recommendation
    recommendation = generate_recommendation(query, context, distances)

    return {
        "Context": context,
        "Sentiment": sentiment,
        "Priority": priority,
        "Recommendation": recommendation
    }

# Step 8: Gradio interface
def gradio_app(query):
    result = process_grievance(query)
    output = (
        f"**Context:**\n{result['Context']}\n\n"
        f"**Sentiment:** {result['Sentiment']} (Confidence: {result['Priority']})\n\n"
        f"**Priority Level:** {result['Priority']}\n\n"
        f"**Recommendation:** {result['Recommendation']}"
    )
    return output

with gr.Blocks() as demo:
    gr.Markdown("# AI-Based Grievance Management System (Enhanced RAG)")
    input_query = gr.Textbox(label="Enter Your Grievance", lines=3)
    output_text = gr.Textbox(label="System Response", lines=15)
    submit_button = gr.Button("Submit")
    submit_button.click(gradio_app, inputs=input_query, outputs=output_text)

demo.launch()