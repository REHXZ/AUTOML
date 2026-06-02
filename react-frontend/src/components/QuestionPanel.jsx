import { Loader2, Send } from "lucide-react";

import { Btn } from "../ui";

export default function QuestionPanel({ questions, answers, onAnswer, onSubmit, loading }) {
  return (
    <section className="questions-panel">
      <div className="section__title">
        <span className="section__title-text">input needed</span>
      </div>
      {questions.map((question, index) => (
        <div key={`${question.question}-${index}`} className="question-item">
          <p>{question.question}</p>
          {question.explanation ? (
            <p className="question-item__explain">{question.explanation}</p>
          ) : null}
          {question.alternatives?.length ? (
            <div className="alternatives">
              {question.alternatives.map((alternative) => (
                <button
                  key={alternative}
                  type="button"
                  onClick={() => {
                    const next = [...answers];
                    next[index] = alternative;
                    onAnswer(next);
                  }}
                >
                  {alternative}
                </button>
              ))}
            </div>
          ) : null}
          <textarea
            className="field"
            value={answers[index] ?? ""}
            onChange={(event) => {
              const next = [...answers];
              next[index] = event.target.value;
              onAnswer(next);
            }}
          />
        </div>
      ))}
      <div style={{ marginTop: 10 }}>
        <Btn variant="primary" size="md" icon={loading ? Loader2 : Send} onClick={onSubmit} disabled={loading}>
          submit answers
        </Btn>
      </div>
    </section>
  );
}
