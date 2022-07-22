using Documenter, OneHotArrays

DocMeta.setdocmeta!(OneHotArrays, :DocTestSetup, :(using OneHotArrays); recursive = true)

makedocs(sitename = "OneHotArrays.jl",
         doctest = false,
         pages = ["Overview" => "index.md",
                  "Reference" => "reference.md"],
         format = Documenter.HTML(
              canonical = "https://fluxml.ai/OneHotArrays.jl/stable/",
              # analytics = "UA-36890222-9",
              assets = ["assets/flux.css"],
              prettyurls = get(ENV, "CI", nothing) == "true"
         ),
)

deploydocs(repo = "github.com/FluxML/OneHotArrays.jl.git",
           target = "build",
           push_preview = true)
