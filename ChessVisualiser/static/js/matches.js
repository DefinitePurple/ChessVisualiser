$(document).ready(function() {

    let cfg = {
        draggable: false,
        position: 'start'
    };
    for(let i=1; i <= 10; i++){
        let board = ChessBoard('board'+i, 'start');
    }
});