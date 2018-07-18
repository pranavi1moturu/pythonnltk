
##Start your code with these libraries:
library(RCurl)
## Loading required package: bitops
library(XML)
library(stringr)
library(httr)

htmlToText <- function(input, ...) {
  ###---PACKAGES ---###
  require(RCurl)
  require(XML)
  
  ###--- LOCAL FUNCTIONS ---###
  # Determine how to grab html for a single input element
  evaluate_input <- function(input) {  
    # if input is a .html file
    if(file.exists(input)) {
      char.vec <- readLines(input, warn = FALSE)
      return(paste(char.vec, collapse = ""))
    }
    
    # if input is html text
    if(grepl("</html>", input, fixed = TRUE)) return(input)
    
    # if input is a URL, probably should use a regex here instead?
    if(!grepl(" ", input)) {
      # downolad SSL certificate in case of https problem
      if(!file.exists("cacert.perm")) download.file(url="http://curl.haxx.se/ca/cacert.pem", destfile="cacert.perm")
      return(getURL(input, followlocation = TRUE, cainfo = "cacert.perm"))
    }
    
    # return NULL if none of the conditions above apply
    return(NULL)
  }
  
  # convert HTML to plain text
  convert_html_to_text <- function(html) {
    doc <- htmlParse(html, asText = TRUE)
    text <- xpathSApply(doc, "//text()[not(ancestor::script)][not(ancestor::style)][not(ancestor::noscript)][not(ancestor::form)]", xmlValue)
    return(text)
  }
  
  # format text vector into one character string
  collapse_text <- function(txt) {
    return(paste(txt, collapse = " "))
  }
  
  ###--- MAIN ---###
  # STEP 1: Evaluate input
  html.list <- lapply(input, evaluate_input)
  
  # STEP 2: Extract text from HTML
  text.list <- lapply(html.list, convert_html_to_text)
  
  # STEP 3: Return text
  text.vector <- sapply(text.list, collapse_text)
  return(text.vector)
}

###Run the function code for htmlToText()(Be sure this function is listed in your Environment)

###Load the first element in the frontier to an "exploredlink" variable

frontier <- c("http://www.cnn.com","http://www.kdnuggets.com","http://news.google.com")

topicwords<-c("technology","school","web","mining","news")

num <- 50 #total number of items to crawl
result <- c()
j <- NULL  #number of items in the repository

while(length(j) < num)
{
  if(length(frontier)<1){
    break
  }
  
  #grab the first item in the frontier and place in the "exploredlink" variable
  exploredlink<-frontier[1]
  frontier<-frontier[-1]
  
  if(str_detect(exploredlink,"\\.jpg$"))
  {
    next
  }
  
  doc <- tryCatch(getURL(exploredlink),error=function(cond){return("")})
  if(str_length(doc)<10){
    next
  }
  domain<-str_extract(exploredlink,pattern = ".*\\.com")
  
  if(is.na(domain)){
    next
  }
  doc <- htmlParse(doc)
  title <- xmlToDataFrame(nodes = getNodeSet(doc, "//title"))
  title <- as.vector(title$text)
  title <- unique(title)
  bodyText<-tryCatch(htmlToText(content(GET(exploredlink),type="text/html",as="text")),error=function(cond){return("")})
  #bodyText<-str_split(tolower(str_replace_all((str_replace_all(bodyText,"(\\t|\\n|\\r)"," ")),"\\s{2,}"," "))," ")[[1]]
  if (any(str_detect(tolower(bodyText),topicwords))){
    j <- append(j,frontier[1])
    result<-append(result,title)
  }
  anchor <- getNodeSet(doc, "//a")
  anchor <- sapply(anchor, function(x) xmlGetAttr(x, "href"))
  
  if(length(anchor)>0){
    temp <- c()
    for(i in 1:length(anchor)){
      if(is.null(anchor[[i]])){
        next
      }
      if(!str_detect(anchor[[i]][1],"^http")){
        next
      }
      temp <- append(temp,str_trim(anchor[[i]][1]))
    }
    anchor <- temp
    rm(temp)
  }
  frontier<-append(frontier,anchor)
}
result=data.frame(j,result)
print(result)
print('Results stored to file crawler_results.tsv')
write.csv(result, file='crawler_results.tsv', row.names=FALSE, na="NULL")

