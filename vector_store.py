"""
============================================================
vector_store.py — ChromaDB Vector Database Manager
============================================================
PURPOSE:
    1. Creates or loads a ChromaDB database on disk
    2. Converts text chunks into embeddings (number vectors)
    3. Stores those embeddings in ChromaDB
    4. Retrieves the most relevant chunks for a question

WHAT IS AN EMBEDDING?
    An embedding converts text into a list of numbers (a vector)
    that captures the MEANING of the text. Similar meanings
    produce similar numbers, so we can find related content
    by comparing these number lists.

    Example (simplified):
        "cat" -> [0.2, 0.8, 0.1, ...]
        "kitten" -> [0.21, 0.79, 0.12, ...]  (very similar!)
        "car" -> [0.9, 0.1, 0.7, ...]        (very different)

WHAT IS ChromaDB?
    ChromaDB is a database that stores these number vectors
    and can quickly find which stored vectors are most similar
    to a new one. It's like a smart search engine that
    understands meaning, not just keywords.

DATABASE PERSISTENCE:
    ChromaDB saves its data to the 'chroma_db/' folder.
    When you restart the program, it loads from disk
    instead of re-processing the PDF. This saves time!
============================================================
"""

import os
import logging
import chromadb
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    TOP_K_RESULTS,
)

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Manages the ChromaDB vector database for our RAG system.

    Think of this class as a smart filing cabinet:
    - You put in documents (chunks of text)
    - It converts each one into a number-code (embedding)
    - When you ask a question, it finds the most relevant docs
    """

    def __init__(self):
        """
        Sets up the vector store when you create an instance.

        WHAT HAPPENS HERE:
        1. Loads the sentence-transformers model (downloads
           ~80MB the first time, then caches it locally)
        2. Opens or creates the ChromaDB database folder
        3. Gets or creates our named collection inside it
        """
        # --------------------------------------------------
        # Step 1: Load the embedding model
        # --------------------------------------------------
        # SentenceTransformer loads a pre-trained AI model
        # that can convert any text into a 384-number vector.
        # This model runs locally — no API key needed!
        # --------------------------------------------------
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model loaded successfully")

        # --------------------------------------------------
        # Step 2: Open ChromaDB with disk persistence
        # --------------------------------------------------
        # PersistentClient saves data to the specified folder.
        # Next time you run the program, it loads from disk
        # instead of recalculating everything.
        # --------------------------------------------------
        logger.info("Connecting to ChromaDB at: %s", CHROMA_PERSIST_DIR)
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

        # --------------------------------------------------
        # Step 3: Get or create our collection
        # --------------------------------------------------
        # get_or_create_collection: if it already exists on
        # disk, it loads it; if not, it creates a new empty one.
        # --------------------------------------------------
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
            # "cosine" means we measure similarity by the angle
            # between vectors. Cosine similarity = 1 means
            # identical meaning, 0 means completely unrelated.
        )
        logger.info(
            "Collection '%s' ready. Contains %d documents.",
            COLLECTION_NAME,
            self.collection.count()
        )

    def is_populated(self) -> bool:
        """
        Checks if the database already has chunks stored.

        Returns:
            True if there are documents in the collection,
            False if the collection is empty.
        """
        count = self.collection.count()
        logger.info("Collection has %d documents", count)
        return count > 0

    def add_chunks(self, chunks: list[dict]) -> None:
        """
        Converts text chunks into embeddings and stores them.

        WHAT HAPPENS:
        1. Takes the text from each chunk
        2. Runs it through the embedding model to get vectors
        3. Stores vectors + metadata + text in ChromaDB

        Args:
            chunks: List of dicts from pdf_processor.chunk_text().
                Each dict has: chunk_id, page, text
        """
        if not chunks:
            logger.warning("No chunks to add!")
            return

        # --------------------------------------------------
        # Prepare the data for ChromaDB
        # --------------------------------------------------
        # ChromaDB needs three parallel lists:
        #   ids:        unique name for each chunk
        #   documents:  the actual text content
        #   metadatas:  extra info (like page numbers)
        # --------------------------------------------------
        ids = [chunk["chunk_id"] for chunk in chunks]
        documents = [chunk["text"] for chunk in chunks]
        metadatas = [{"page": chunk["page"]} for chunk in chunks]

        # --------------------------------------------------
        # Generate embeddings for all chunks at once
        # --------------------------------------------------
        # model.encode() converts all texts into vectors.
        # Doing them all at once (batch) is much faster than
        # doing them one by one.
        # --------------------------------------------------
        logger.info("Generating embeddings for %d chunks...", len(chunks))
        embeddings = self.embedding_model.encode(
            documents,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # Convert numpy arrays to plain Python lists
        # (ChromaDB expects regular lists, not numpy arrays)
        embeddings_list = [emb.tolist() for emb in embeddings]

        # --------------------------------------------------
        # Store everything in ChromaDB
        # --------------------------------------------------
        # We use upsert instead of add:
        #   add() = fails if the ID already exists
        #   upsert() = adds if new, updates if existing
        # This makes the code safe to run multiple times.
        # --------------------------------------------------
        logger.info("Storing %d chunks in ChromaDB...", len(chunks))
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings_list
        )
        logger.info("Successfully stored all chunks in the database!")

    def query(self, question: str, top_k: int = TOP_K_RESULTS) -> list[dict]:
        """
        Finds the most relevant chunks for a given question.

        HOW IT WORKS:
        1. Converts the question text into an embedding vector
        2. Asks ChromaDB: "which stored vectors are most
           similar to this question vector?"
        3. ChromaDB returns the top_k closest matches

        Args:
            question: The user's question as a string.
            top_k: How many results to return (default: 5).

        Returns:
            List of dicts, each containing:
                - chunk_id: The unique ID of the chunk
                - page: Which PDF page it came from
                - text: The actual text content
                - distance: How different from the question
                  (lower = more similar, 0 = exact match)
        """
        # --------------------------------------------------
        # Step 1: Convert question to an embedding
        # --------------------------------------------------
        question_embedding = self.embedding_model.encode(
            [question],
            convert_to_numpy=True
        )
        question_embedding_list = question_embedding[0].tolist()

        # --------------------------------------------------
        # Step 2: Search ChromaDB for similar chunks
        # --------------------------------------------------
        results = self.collection.query(
            query_embeddings=[question_embedding_list],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        # --------------------------------------------------
        # Step 3: Format the results into clean dicts
        # --------------------------------------------------
        # ChromaDB returns nested lists (because you could
        # query multiple questions at once). We only have
        # one question, so we take index [0] everywhere.
        # --------------------------------------------------
        formatted_results = []

        if results and results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                formatted_results.append({
                    "chunk_id": results["ids"][0][i],
                    "page": results["metadatas"][0][i]["page"],
                    "text": results["documents"][0][i],
                    "distance": results["distances"][0][i]
                })

        logger.info(
            "Query returned %d results for: '%s'",
            len(formatted_results),
            question[:50]
        )
        return formatted_results
