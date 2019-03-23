function hide(){
      const style = {
        height: '0',
        fontSize: '0'
      };
      $("div.flash").animate(style, function () {
        $(this).remove()
      });
    }

    $(document).ready(function(){
      window.setInterval(hide, 4000);
    });