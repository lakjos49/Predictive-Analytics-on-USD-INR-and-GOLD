document.addEventListener("DOMContentLoaded", () => {
    const backendUrl = "http://127.0.0.1:5000";

    const singleDateInput = document.getElementById("single-date-input");
    const predictButton = document.getElementById("predict-button");
    const predictionResults = document.getElementById("prediction-results");
    const loadingSpinnerSingle = document.getElementById("loading-spinner-single");

    const startDateInput = document.getElementById("start-date-input");
    const endDateInput = document.getElementById("end-date-input");
    const plotButton = document.getElementById("plot-button");
    const plotContainer = document.getElementById("plot-container");
    const loadingSpinnerPlot = document.getElementById("loading-spinner-plot");

    // Handle Point Prediction
    predictButton.addEventListener("click", async () => {
        const date = singleDateInput.value;
        if (!date) {
            predictionResults.innerHTML = '<p class="text-red-400">Please select a date.</p>';
            return;
        }

        loadingSpinnerSingle.classList.remove("hidden");
        predictionResults.innerHTML = "";

        try {
            const response = await fetch(`${backendUrl}/predict_point`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ date: date }),
            });

            const result = await response.json();

            if (response.ok) {
                predictionResults.innerHTML = `
                    <p class="text-lg font-semibold text-white">Predicted 24K Gold Rate: INR ${result.gold_rate}</p>
                    <p class="text-lg font-semibold text-white">Predicted USD/INR Rate: ${result.inr_usd_rate}</p>
                `;
            } else {
                predictionResults.innerHTML = `<p class="text-red-400">Error: ${result.error || "Prediction failed"}</p>`;
            }
        } catch (e) {
            console.error("Error during API call:", e);
            predictionResults.innerHTML = '<p class="text-red-400">An error occurred. Check the console for details.</p>';
        } finally {
            loadingSpinnerSingle.classList.add("hidden");
        }
    });

    // Handle Plot Generation
    plotButton.addEventListener("click", async () => {
        const startDate = startDateInput.value;
        const endDate = endDateInput.value;

        if (!startDate || !endDate) {
            plotContainer.innerHTML = '<p class="text-red-400">Please select both start and end dates.</p>';
            return;
        }

        loadingSpinnerPlot.classList.remove("hidden");
        plotContainer.innerHTML = ""; // Clear previous plot

        try {
            const response = await fetch(`${backendUrl}/predict_range`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    start_date: startDate,
                    end_date: endDate,
                }),
            });

            if (
                response.headers.get("content-type") &&
                response.headers.get("content-type").includes("application/json")
            ) {
                const result = await response.json();

                if (response.ok) {
                    const imageBase64 = result.plot_image;
                    if (imageBase64) {
                        plotContainer.innerHTML = `<img src="data:image/png;base64,${imageBase64}" alt="Predicted rates plot" class="w-full h-full object-contain rounded-lg">`;
                    } else {
                        plotContainer.innerHTML = `<p class="text-red-400">Error: Plot image data missing from response.</p>`;
                    }
                } else {
                    plotContainer.innerHTML = `<p class="text-red-400">Server Error: ${
                        result.error || "Plot generation failed."
                    }</p>`;
                    console.error("Backend server error:", result.error);
                }
            } else {
                const text = await response.text();
                plotContainer.innerHTML = `<p class="text-red-400">Invalid response from server. Check the backend log for errors.</p>`;
                console.error("Non-JSON response from server:", text);
            }
        } catch (e) {
            console.error("Error during API call:", e);
            plotContainer.innerHTML = `<p class="text-red-400">An error occurred: ${e.message}. Check the console for details.</p>`;
        } finally {
            loadingSpinnerPlot.classList.add("hidden");
        }
    });
});
