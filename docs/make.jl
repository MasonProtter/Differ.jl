using Documenter
using DocumenterCodeBlocks
using Differ, Contextual, DifferCore, DifferForwards, DifferReverse

makedocs(;
    sitename = "Differ.jl",
    modules = [Differ, Contextual, DifferCore, DifferForwards, DifferReverse],
    plugins = [CodeBlocks()],
    doctest = true,
    pages = [
        "Home" => "index.md",
        "Writing rules" => "custom_rules.md",
        "API Reference" => "reference.md",
    ],
)

deploydocs(;
    repo = "github.com/MasonProtter/Differ.jl.git",
    devbranch = "master",
    push_preview = true,
)
