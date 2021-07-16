using DataFrames, Query
function counter(df::DataFrame, col::Symbol)
    for level in unique(df[!, col])
        t = df |>
            @filter(_[col] == level)|>
            DataFrame
        len_t = length(t[!, col])
        per_t = round(100*length(t[!, col])/length(df[!, col]), digits=2)
        println("$level: $len_t ($per_t% of total)")
    end
end

