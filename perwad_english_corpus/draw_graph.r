data = read.csv("tmp/lemma_countx.txt", header=F)

png(filename="10K_lemmas.png")
plot(data,type="l", xlab="Number of lemmas", ylab="Percentage of corpus", col="blue", main="10000 Most Common Lemmas")

png(filename="1K_lemmas.png")
plot(head(data,100),type="l", xlab="Number of lemmas", ylab="Percentage of corpus", col="blue", main="1000 Most Common Lemmas")
dev.off()
