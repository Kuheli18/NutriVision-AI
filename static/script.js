const input = document.getElementById("foodImage");
const zone = document.getElementById("dropzone");
const previewWrap = document.getElementById("previewWrap");
const preview = document.getElementById("preview");
const btn = document.getElementById("analyzeBtn");
const form = document.getElementById("uploadForm");
const statusEl = document.getElementById("status");
const results = document.getElementById("results");

let chosen = null;
let nutrition = null;
let previewUrl = null;


// STATUS MESSAGE
function status(msg, error = false) {
    statusEl.textContent = msg;
    statusEl.className = error ? "status error" : "status";
}


// SELECT IMAGE
function choose(file) {
    if (!file) return;

    if (!file.type.startsWith("image/")) {
        return status("Please select a valid image.", true);
    }

    if (file.size > 10 * 1024 * 1024) {
        return status("Image must be smaller than 10 MB.", true);
    }

    chosen = file;

    if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
    }
    previewUrl = URL.createObjectURL(file);
    preview.src = previewUrl;

    previewWrap.hidden = false;
    zone.hidden = true;

    btn.disabled = false;
    btn.textContent = "Analyze my food →";

    results.hidden = true;

    status("");
}


// IMAGE INPUT
input.addEventListener("change", e => {
    choose(e.target.files[0]);
});


// DRAG AND DROP
["dragenter", "dragover"].forEach(event => {
    zone.addEventListener(event, e => {
        e.preventDefault();
        zone.classList.add("drag");
    });
});

["dragleave", "drop"].forEach(event => {
    zone.addEventListener(event, e => {
        e.preventDefault();
        zone.classList.remove("drag");
    });
});

zone.addEventListener("drop", e => {
    choose(e.dataTransfer.files[0]);
});


// REMOVE IMAGE
document.getElementById("remove").addEventListener("click", () => {

    chosen = null;

    if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
        previewUrl = null;
    }

    input.value = "";
    preview.src = "";

    previewWrap.hidden = true;
    zone.hidden = false;

    btn.disabled = true;
    btn.textContent = "Analyze my food →";

    results.hidden = true;

    status("");
});


// ANALYZE IMAGE
form.addEventListener("submit", async e => {

    e.preventDefault();

    if (!chosen) return;

    const data = new FormData();
    data.append("food_image", chosen);

    btn.disabled = true;
    btn.textContent = "Analyzing...";

    status("Analyzing your image. Please wait...");

    try {

        const res = await fetch("/analyze", {
            method: "POST",
            body: data
        });

        const out = await res.json();

        if (!res.ok) {
            throw new Error(out.error || "Analysis failed");
        }

        show(out);

        status("Analysis completed successfully.");

    } catch (err) {

        status(err.message, true);

    } finally {

        btn.disabled = false;
        btn.textContent = "Analyze my food →";

    }

});


// DISPLAY RESULTS
function show(d) {

    results.hidden = false;

    // FOOD IDENTIFICATION
    document.getElementById("foodName").textContent = d.food_label;
    document.getElementById("foodScore").textContent = d.score + "%";


    // DETECTION TYPE
    let predictionBox = document.getElementById("detectionType");

    if (!predictionBox) {

        predictionBox = document.createElement("p");
        predictionBox.id = "detectionType";
        predictionBox.className = "muted";

        document.getElementById("foodName")
            .after(predictionBox);

    }

    predictionBox.textContent = d.detection_type || "Fine-Tuned ResNet34 (Food-101)";


    // TOP PREDICTIONS
    const p = document.getElementById("predictions");

    p.innerHTML = "";

    d.top_predictions.forEach(x => {

        const row = document.createElement("div");
        row.className = "prediction";

        const name = document.createElement("span");
        const score = document.createElement("b");

        name.textContent = x.label;
        score.textContent = x.score + "%";

        row.append(name, score);

        p.append(row);

    });


    // MULTIPLE FOOD DETECTIONS
    let detectedBox = document.getElementById("detectedItems");

    if (!detectedBox) {

        detectedBox = document.createElement("div");
        detectedBox.id = "detectedItems";
        detectedBox.className = "detected-items";

        document.getElementById("predictions")
            .after(detectedBox);

    }

    detectedBox.innerHTML = "";

    if (d.detected_items && d.detected_items.length > 0) {

        const heading = document.createElement("h4");
        heading.textContent = "Detected food items";

        detectedBox.append(heading);

        d.detected_items.forEach(item => {

            const row = document.createElement("div");
            row.className = "prediction";

            const name = document.createElement("span");
            const details = document.createElement("b");

            name.textContent =
                item.label + " × " + item.count;

            details.textContent =
                item.score + "%";

            row.append(name, details);

            detectedBox.append(row);

        });

    }


    // NUTRITION INFORMATION
    nutrition = d.nutrition_per_100g;

    const intro = document.getElementById("nutritionIntro");
    const wrap = document.getElementById("servingWrap");

    if (nutrition) {

        intro.textContent =
            "Reference values for " +
            d.nutrition_name +
            " per 100 g";

        wrap.hidden = false;

        calc();

    } else {

        intro.textContent =
            "No matching nutrition reference is available for this predicted category.";

        wrap.hidden = true;

        document.getElementById("nutritionGrid").innerHTML = "";

    }


    // IMAGE QUALITY
    document.getElementById("lighting").textContent =
        d.image_quality.lighting;

    document.getElementById("focus").textContent =
        d.image_quality.focus;

    document.getElementById("brightness").textContent =
        d.image_quality.brightness + " / 255";

    document.getElementById("sharpness").textContent =
        d.image_quality.sharpness;


    // SCROLL TO RESULTS
    results.scrollIntoView({
        behavior: "smooth"
    });

}


// NUTRITION CALCULATOR
function calc() {

    if (!nutrition) return;

    const g = Number(
        document.getElementById("serving").value
    );

    document.getElementById("grams").textContent = g;

    const f = g / 100;
    const fibrePer100g = Number(nutrition.fibre ?? nutrition.fiber ?? 0);

    const items = [
        ["Calories", Math.round(nutrition.calories * f) + " kcal"],
        ["Protein", (nutrition.protein * f).toFixed(1) + " g"],
        ["Carbs", (nutrition.carbs * f).toFixed(1) + " g"],
        ["Fat", (nutrition.fat * f).toFixed(1) + " g"],
        ["Fibre", (fibrePer100g * f).toFixed(1) + " g"]
    ];

    const grid = document.getElementById("nutritionGrid");

    grid.innerHTML = "";

    items.forEach(([name, value]) => {

        const card = document.createElement("div");
        card.className = "nutrient";

        const title = document.createElement("span");
        const amount = document.createElement("strong");

        title.textContent = name;
        amount.textContent = value;

        card.append(title, amount);

        grid.append(card);

    });

}


// SERVING SIZE
document.getElementById("serving")
    .addEventListener("input", calc);


// DARK AND LIGHT THEME
const toggle = document.getElementById("themeToggle");

const theme =
    localStorage.getItem("nutrivision-theme") || "light";

document.documentElement.dataset.theme = theme;

toggle.textContent =
    theme === "dark" ? "☀" : "☾";


toggle.addEventListener("click", () => {

    const next =
        document.documentElement.dataset.theme === "dark"
            ? "light"
            : "dark";

    document.documentElement.dataset.theme = next;

    localStorage.setItem("nutrivision-theme", next);

    toggle.textContent =
        next === "dark" ? "☀" : "☾";

});