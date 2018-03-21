
// Status
function loading_overlay(count){
        $.LoadingOverlay("progress", count);
        $.LoadingOverlay("text", "Working...");
        }

function pending_overlay(){
  $.LoadingOverlay("show", {
          image       : "",
          text        : "Pending...",
          progress    : true
          });
}

function checkExists(sel) {
    var status = false;
    if ($(sel).length) status = true;
    return status;
}

//Program a custom submit function for the form
$("form#{{ form_name  }}").submit(function(event){

  //disable the default form submission
  event.preventDefault();

  //grab all form data
  var formData = new FormData($(this)[0]);

  $.ajax({
    url: "{{ url }}",
   type: 'POST',
    data: formData,
    async: true,
    cache: false,
    contentType: false,
    processData: false,
    dataType: 'json',
    success: function(data, textStatus,jqXHR) {
                    status_url = jqXHR.getResponseHeader('Location');
                    redirect_url = jqXHR.getResponseHeader('Redirect');

                    update_progress(status_url,redirect_url);
                },
    error: function(jqXHR) {
        //<div class="alert alert-danger" id="flash_alert">
        //error_message = data['error_message']//jqXHR.getResponseHeader('error_message');
        //</div>

        var error_message = jqXHR.responseJSON.message

        if (checkExists('#alert_list')) {
            // alert('alert_list');
            div = $("".concat('<li>',error_message,'</li>'));
            $('#alert_list').append(div);
        }
        else {
            //alert('flash_alert');
            div = $("".concat('<div class="alert alert-danger" id="flash_alert"> Warning: <ul type="circle" id="alert_list"> <li>',error_message,'</li></ul></div>'));
            $('#flash_alert').append(div);
        }
        //alert(error_message);


        // alert('Unexpected error');
    }
  });

  return false;
});

function update_progress(status_url,redirect_url) {

    pending_overlay();
    // send GET request to status URL
    $.getJSON(status_url, function(data) {
        // update UI
        percent = parseInt(data['current'] * 100 / data['total']);

        if (percent > 0) {
          loading_overlay(percent)
        }
        if (data['state'] != 'PENDING' && data['state'] != 'PROGRESS') {
            if ('result' in data) {
                // show result
                $("html").load(redirect_url);
                $(location).attr('href',redirect_url);
            }
            else {
                // something unexpected happened
                $(status_div.childNodes[3]).text('Result: ' + data['state']);

            }
        }
        else {
            // rerun in x seconds
            setTimeout(function() {
                update_progress(status_url, redirect_url);
            }, {{ timeout }});
        }
    });
}