$(document).ready(function() {
    // Setup string formatting
    if (!String.prototype.format) {
      String.prototype.format = function() {
        var args = arguments;
        return this.replace(/{(\d+)}/g, function(match, number) {
          return typeof args[number] != 'undefined'
            ? args[number]
            : match
          ;
        });
      };
    }


    // Get the parent container
    const parent = $('div#matches-container');
    // Create the new row
    const row = "<div class=\"row mb-4\" id=\"matches-row\"></div>";

    for (let i = 0; i < uploads.length; i++){
        let upload = uploads[i];
        let text =  "White: {0} \n".format(upload.white)+
                    "Black: {0} \n".format(upload.black)+
                    "Score: {0} \n".format(upload.score)+
                    "ID: {0} \n".format(upload.id)+
                    "Upload Date: {0}".format(upload.date);
        let url = "{{ url_for(\"match.view_upload\") }}";

        let html = "<div class=\"d-flex col-2 justify-content-center\">" +
        "<form class=\"card game-card\" style=\"width: 80%;\">" +
        "<div class=\"card-body\">" +
        "<pre class=\"card-text\">{0}</pre>".format(text) +
        "<a href=\"{{url_for('math.view_upload' }}\" class=\"btn btn-primary\">View</a>" +
        "</div>" +
        "</form>" +
        "</div>";

        console.log(html);
        if (i%6 === 0){
            $('div#matches-row').removeAttr('id');
            parent.append(row);
        }
        $('div#matches-row').append(html)

    }
});