document.getElementById("searchBtn").addEventListener("click", async () => {
  const keyword = document.getElementById("keyword").value; // FIXED
  const resultsDiv = document.getElementById("results");
  const errorDiv = document.getElementById("error");

  resultsDiv.innerHTML = "";
  errorDiv.textContent = "";

  if (!keyword.trim()) {
    errorDiv.textContent = "Please enter a keyword.";
    return;
  }

  try {
    const response = await fetch("http://127.0.0.1:8000/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ keyword }) // FIXED
    });

    if (!response.ok) throw new Error(`Server returned ${response.status}`);
    const data = await response.json();
    resultsDiv.innerHTML = `<h3>${data.session_title}</h3><p>${data.summary}</p>`;
  } catch (err) {
    errorDiv.textContent = `Error: ${err.message}. Ensure backend is running.`;
  }
});
