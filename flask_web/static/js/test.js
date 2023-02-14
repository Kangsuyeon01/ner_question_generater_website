
var number_id, problem_id, answer_id, ox_question_id,ox_answer_id,wrong_answer_id;
var problem_arr;
var answer_arr;

var e_q; // 빈칸문제
var e_a; // 빈칸 문제 답
var ox_q; // OX 문제
var ox_a; // OX 문제 답
var my_answer_arr = []; // 내가 입력한 답
var selectAnswer, score;
var imgNum, startNum, endNum;

function data_load(data){
  var tmp_data = eval(data);

  for (var i in tmp_data){
    if (i == 0){
      e_q = tmp_data[0];
    }
    else if (i == 1){
      e_a = tmp_data[1];
    }
    else if (i == 2){
      ox_q = tmp_data[2];
    }
    else if (i == 3){
      ox_a = tmp_data[3];      
    }

    
}
return e_q,e_a,ox_q,ox_a;
}


function init(){
  number_id = document.getElementById("number_id");
  problem_id = document.getElementById("problem_id");
  answer_id = document.getElementById("answer_id");
  questions_id = document.getElementById("questions_id").innerText;
  wrong_answer_id = document.getElementById("wrong_answer_id");
  e_q,e_a,ox_q,ox_a = data_load(questions_id);

  problem_arr = ox_q;
  answer_arr = ox_a;
  console.log(answer_arr);

  startNum = 0;
  endNum = ox_q.length;
  score = 0;
  setProblem();
  
}


init();

function setProblem(){
  problem_id.innerHTML = "<span style=\"font: 25px / 1.6 kdm;\">" + problem_arr[startNum] + "</span>";
  if(startNum == endNum){
    number_id.innerHTML = "<span class='label'>< 결과보기 ></span>";
    problem_id.innerHTML = "<h2>" + "모든 문제를 풀었습니다." + "</h2>";
    answer_id.innerHTML = "<button type='button' class='button o' onclick='btnResFunc();'>결과보기</button><button type='button' class='button o' onclick='btnWrong();'>틀린 문제</button>";
    wrong_answer_id.innerHTML = "<button type='button' class='button o' onclick='history.go(0);'>다시하기</button>"
  } else {
    
    number_id.innerHTML = "<span class='label'>< " + parseInt(startNum + 1) + " ></span>";
    answer_id.innerHTML = "<button type='button' class='button o' onclick='btnOFunc();'>O</button><button type='button' class='button x' onclick='btnXFunc();'>X</button>";
  }
  
}

function btnOFunc(){
  selectAnswer = "O";
  if(answer_arr[startNum][0] == selectAnswer){
    score++;
  }
  else {
    my_answer_arr.push(startNum);
  }
  startNum++;
  setProblem();
}

function btnXFunc(){
  selectAnswer = "X";
  if(answer_arr[startNum][0] == selectAnswer){
    score++;
  }
  else {
    my_answer_arr.push(startNum);
  }
  startNum++;
  setProblem();
}
function btnWrong(){
  
  tmp_arr = ''
  
  for (var i in my_answer_arr){
    tmp_arr += "<span style=\"font: 80% / 1.6 kdm;\">"+ "Q" + i + ". " + problem_arr[my_answer_arr[i]]+ "</span>" + "<br>"; 
    tmp_arr += "<span style=\"font: 80% / 1.6 kdm;\">"+ "A" + i + ". " + answer_arr[my_answer_arr[i]]+ "</span>" + "<br>"; 
  }
  wrong_answer_id.innerHTML = "<button type='button' class='button o' onclick='history.go(0);'>다시하기</button>"
  answer_id.innerHTML = "<button type='button' class='button o' onclick='btnResFunc();'>결과보기</button>"
  problem_id.innerHTML = tmp_arr;

  number_id.innerHTML = "<span class='label'>< 틀린 문제 ></span>";
}
function wrong_index(ox_q){
  n = ox_q.length;
  var wrong_arr = new Array(n);
  return wrong_arr;
}





function btnResFunc(){
  Swal.fire({
    title: '',
    text: '',
    html: "<b>" + endNum +"개의 문제 중에 "+ score+"개를 맞추었습니다.</b> <br><b>당신의 점수는 " + (score / endNum) * 100  + "점입니다.</b>",
    icon: 'success',
    confirmButtonColor: '#d33',
    confirmButtonText: '닫기',
    allowOutsideClick: false
  })
}

