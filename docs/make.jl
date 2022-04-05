using Documenter, OneHotArrays

DocMeta.setdocmeta!(OneHotArrays, :DocTestSetup, :(using OneHotArrays); recursive = true)
make(sitename = "OneHotArrays", doctest = false,
     pages = ["Overview" => "index.md",
              "Reference" => "reference.md"])

deploydocs(repo = "github.com/FluxML/OneHotArrays.jl.git",
           target = "build",
           push_preview = true)
