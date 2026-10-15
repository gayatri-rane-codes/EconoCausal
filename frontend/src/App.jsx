import React, { useState } from 'react';
import Header from './components/Header';
import BudgetInput from './components/BudgetInput';
import DataUpload from './components/DataUpload';
import DataSummary from './components/DataSummary';
import './App.css';

export default function App() {
  const [budget, setBudget] = useState('5000');
  const [budgetError, setBudgetError] = useState('');
  const [fileInfo, setFileInfo] = useState(null);
  const [uploadError, setUploadError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  return (
    <div className="app-container">
      <Header />
      <main className="main-content">
        <BudgetInput
          budget={budget}
          setBudget={setBudget}
          budgetError={budgetError}
          setBudgetError={setBudgetError}
        />
        <DataUpload
          fileInfo={fileInfo}
          setFileInfo={setFileInfo}
          uploadError={uploadError}
          setUploadError={setUploadError}
          isLoading={isLoading}
          setIsLoading={setIsLoading}
        />
        <DataSummary fileInfo={fileInfo} />
      </main>
      <footer className="app-footer">
        <p>EconoCausal · Phase 1 Foundation · Client-Side Inspection Only</p>
      </footer>
    </div>
  );
}
